/******************************************************************************
 *  CKKS 向量模糊匹配（Cosine，L2 归一化，Diagonal + BSGS）
 *
 *  本版新增：复槽 real+imag 双 embedding 打包的 BSGS 版本
 *
 *  典型设置：
 *    - poly_modulus_degree   = 4096
 *    - NUM_EMBEDDINGS        = 4096
 *    - DIM                   = 1024（要求 DIM | slot_count）
 *    - OpenMP 并行
 *
 *  CKKS 参数（建议）：
 *    - CKKS_SCALE             = 2^30
 *    - coeff_modulus          = { 40, 29, 40 }   // 单层 cipher×plain 深度
 *    - 每次 multiply_plain 后：rescale_to_next
 *    - add 前通过 mod_switch_to + 对齐 scale（add_aligned_inplace）
 *
 *  提供三个接口：
 *    1) vector_match_ckks_naive_diag    —— 朴素对角线（1024 次 rotate）
 *    2) vector_match_ckks_bsgs          —— 原始 BSGS（可留作 baseline）
 *    3) vector_match_ckks_bsgs_packed   —— ★ 复槽打包两条 embedding 的 BSGS
 ******************************************************************************/

#include "seal/seal.h"
#include <omp.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace std;
using namespace seal;

/* ------------------- 通用结构 & 工具 ------------------- */

enum class EmbMode { Gaussian, Sparse, Uniform, Test };
enum class Metric  { Cosine };

struct BSGSPlan {
    size_t baby;
    size_t giant;
};

static inline BSGSPlan plan_bsgs(size_t dim) {
    size_t b = static_cast<size_t>(floor(sqrt(static_cast<double>(dim))));
    b = max<size_t>(1, b);
    size_t g = (dim + b - 1) / b;
    return { b, g };
}

/* embedding 生成：和你当前版本保持一致 */
vector<vector<double>> generate_embeddings_ckks(
    size_t num_embeddings,
    size_t dim,
    EmbMode mode = EmbMode::Gaussian,
    uint32_t seed = std::random_device{}(),
    double scale = 1.0,
    double sparsity = 0.9,
    bool l2_normalize = true
) {
    std::mt19937 rng(seed);
    vector<vector<double>> emb(num_embeddings, vector<double>(dim, 0.0));

    if (mode == EmbMode::Gaussian) {
        vector<double> dim_std(dim);
        for (size_t d = 0; d < dim; ++d) {
            double frac = static_cast<double>(d + 1) / dim;
            dim_std[d] = 0.5 + 1.5 * frac; // [0.5, 2.0]
        }
        normal_distribution<double> N01(0.0, 1.0);
        for (size_t i = 0; i < num_embeddings; ++i) {
            for (size_t d = 0; d < dim; ++d) {
                emb[i][d] = N01(rng) * dim_std[d] * scale;
            }
        }
    } else if (mode == EmbMode::Sparse) {
        normal_distribution<double> N(0.0, 1.0);
        bernoulli_distribution keep(1.0 - sparsity);
        for (size_t i = 0; i < num_embeddings; ++i) {
            for (size_t d = 0; d < dim; ++d) {
                if (keep(rng)) emb[i][d] = N(rng) * scale;
            }
        }
    } else if (mode == EmbMode::Test) {
        // Test 模式：全部置 1
        for (size_t i = 0; i < num_embeddings; ++i) {
            std::fill(emb[i].begin(), emb[i].end(), 1.0);
        }
        return emb;
    } else { // Uniform
        uniform_real_distribution<double> U(-1.0, 1.0);
        for (size_t i = 0; i < num_embeddings; ++i) {
            for (size_t d = 0; d < dim; ++d) {
                emb[i][d] = U(rng) * scale;
            }
        }
    }

    if (l2_normalize && mode != EmbMode::Test) {
        for (size_t i = 0; i < num_embeddings; ++i) {
            double norm = 0.0;
            for (size_t d = 0; d < dim; ++d) norm += emb[i][d] * emb[i][d];
            norm = sqrt(norm);
            if (norm == 0.0) continue;
            for (size_t d = 0; d < dim; ++d) emb[i][d] /= norm;
        }
    }
    return emb;
}

/* 明文 Cosine + Top-k */
double cosine_similarity_ckks(const vector<double> &vec1, const vector<double> &vec2) {
    double dot_product = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) dot_product += vec1[i] * vec2[i];
    return dot_product;
}

set<size_t> get_top_k_indices_ckks(const vector<double>& query,
                                   const vector<vector<double>>& db,
                                   size_t k) {
    vector<pair<double, size_t>> similarities(db.size());
    for (size_t i = 0; i < db.size(); ++i) {
        similarities[i] = { cosine_similarity_ckks(query, db[i]), i };
    }
    sort(similarities.rbegin(), similarities.rend());
    set<size_t> top_k_indices;
    for (size_t i = 0; i < k && i < similarities.size(); ++i) {
        top_k_indices.insert(similarities[i].second);
    }
    return top_k_indices;
}

/* 通用 1D 槽向量右移（double / complex 都能用） */
template<typename T>
static inline void shift_slots_cyclic(std::vector<T> &v, size_t sft) {
    size_t n = v.size();
    if (!n) return;
    sft %= n;
    if (!sft) return;
    std::vector<T> tmp(n);
    for (size_t i = 0; i < n; ++i) {
        tmp[(i + sft) % n] = v[i]; // 右移 sft
    }
    v.swap(tmp);
}

/* add 前对齐 level + scale */
inline void add_aligned_inplace(Evaluator &eval,
                                Ciphertext &acc,
                                Ciphertext &term,
                                double target_scale)
{
    if (acc.parms_id() != term.parms_id()) {
        eval.mod_switch_to_inplace(acc, term.parms_id());
    }
    acc.scale()  = target_scale;
    term.scale() = target_scale;
    eval.add_inplace(acc, term);
}

/* 查询向量编码 & 加密：用 CKKS_SCALE，保持“纯实” */
Ciphertext encrypt_query_from_vec_ckks(
    CKKSEncoder &encoder,
    Encryptor &encryptor,
    size_t slot_count,
    size_t dim,
    const std::vector<double> &qvec,
    double scale)
{
    std::vector<double> slots(slot_count, 0.0);
    for (size_t i = 0; i < slot_count; ++i) slots[i] = qvec[i % dim];

    Plaintext pt;
    encoder.encode(slots, scale, pt);

    Ciphertext ct;
    encryptor.encrypt(pt, ct);
    return ct;
}

/* ------------------- 朴素对角线版本（保留做 baseline） ------------------- */

/*
 * block s:
 *   slot i ↔ global embedding index gid = s*slot_count + i
 *
 * 第 d 条「对角线」明文 db[s][d]：
 *   db[s][d][i] = E[gid][ (i + d) % DIM ]
 *
 * 查询向量 q 编码为：
 *   Q[i] = q[i % DIM]（依赖 DIM | slot_count）
 *
 * rotate_vector(Q, d) 左移 d 个槽：
 *   Q_d[i] = q[(i + d) % DIM]
 *
 * Σ_d Q_d[i] * db[s][d][i] = Σ_d q[(i+d)%DIM] * E[gid][(i+d)%DIM]
 *                          = <q, E[gid]>
 */
std::vector<std::vector<Plaintext>> encode_database_diagonal_ckks_naive(
    const SEALContext &context,
    const std::vector<std::vector<double>> &embeddings,
    size_t slot_count,
    size_t dim,
    double scale
) {
    const size_t num_embeddings = embeddings.size();
    const size_t fd = (num_embeddings + slot_count - 1) / slot_count;

    std::vector<std::vector<Plaintext>> db(fd, std::vector<Plaintext>(dim));

    #pragma omp parallel
    {
        CKKSEncoder encoder_local(context);

        #pragma omp for schedule(static)
        for (size_t s = 0; s < fd; ++s) {
            for (size_t d = 0; d < dim; ++d) {
                std::vector<double> slots(slot_count, 0.0);

                for (size_t i = 0; i < slot_count; ++i) {
                    size_t gid = s * slot_count + i;
                    if (gid >= num_embeddings) break;
                    size_t idx = (i + d) % dim;
                    slots[i] = embeddings[gid][idx];
                }

                encoder_local.encode(slots, scale, db[s][d]);
            }
        }
    }
    return db;
}

/* naive：生成所有 Q_d = Rot(Q, d) */
std::vector<Ciphertext> make_queryset_ckks_naive(
    const SEALContext &context,
    const Ciphertext &query,
    const GaloisKeys &galois_keys,
    size_t dim)
{
    std::vector<Ciphertext> qs(dim);
    qs[0] = query;

    #pragma omp parallel
    {
        Evaluator eval_local(context);

        #pragma omp for schedule(static)
        for (size_t d = 1; d < dim; ++d) {
            eval_local.rotate_vector(query, static_cast<int>(d), galois_keys, qs[d]);
        }
    }
    return qs;
}

/* 朴素对角线主流程（与你当前版本一致） */
void vector_match_ckks_naive_diag(
    size_t POLY_DEGREE,
    size_t NUM_EMBEDDINGS,
    size_t DIM,
    size_t OMP_THREADS,
    EmbMode data_mode,
    uint32_t data_seed,
    uint32_t query_seed,
    double CKKS_SCALE,
    size_t TOP_K)
{
    omp_set_num_threads(static_cast<int>(OMP_THREADS));

    bool L2_NORMALIZE = true;

    auto embeddings = generate_embeddings_ckks(
        NUM_EMBEDDINGS, DIM,
        data_mode, data_seed, 1.0, 0.9, L2_NORMALIZE);

    std::vector<double> qvec;
    size_t random_idx = 0;

    if (data_mode == EmbMode::Test) {
        qvec.assign(DIM, 1.0);
    } else {
        std::mt19937 gen(query_seed ? query_seed : std::random_device{}());
        std::uniform_int_distribution<size_t> dis(0, NUM_EMBEDDINGS - 1);
        random_idx = dis(gen);
        qvec = embeddings[random_idx];
    }

    std::set<size_t> true_top_k;
    if (data_mode != EmbMode::Test) {
        true_top_k = get_top_k_indices_ckks(qvec, embeddings, TOP_K);
    }

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(POLY_DEGREE);
    // 单层 cipher×plain 深度：3 个模数足够
    parms.set_coeff_modulus(CoeffModulus::Create(
        POLY_DEGREE, { 40, 29, 40 }));

    SEALContext context(parms);
    if (!context.parameters_set()) {
        throw std::runtime_error("[CKKS-NAIVE] encryption parameters are not valid.");
    }

    KeyGenerator keygen(context);
    PublicKey  pk;  keygen.create_public_key(pk);
    SecretKey  sk = keygen.secret_key();
    GaloisKeys gk;  keygen.create_galois_keys(gk);

    Encryptor   encryptor(context, pk);
    Decryptor   decryptor(context, sk);
    CKKSEncoder encoder(context);

    const size_t slot_count = encoder.slot_count(); // = POLY_DEGREE / 2
    if (DIM > slot_count || (slot_count % DIM) != 0) {
        throw std::invalid_argument(
            "[CKKS-NAIVE] require DIM <= slot_count and slot_count % DIM == 0.");
    }
    const size_t fd = (NUM_EMBEDDINGS + slot_count - 1) / slot_count;

    std::cerr << "[CKKS-NAIVE] poly_degree=" << POLY_DEGREE
              << ", slot_count=" << slot_count
              << ", NUM_EMBEDDINGS=" << NUM_EMBEDDINGS
              << ", DIM=" << DIM
              << ", fd=" << fd << "\n";

    auto db = encode_database_diagonal_ckks_naive(
        context, embeddings, slot_count, DIM, CKKS_SCALE);

    Ciphertext query_ct = encrypt_query_from_vec_ckks(
        encoder, encryptor, slot_count, DIM, qvec, CKKS_SCALE);

    auto queryset = make_queryset_ckks_naive(context, query_ct, gk, DIM);

    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<Ciphertext> result(fd);
    const double TARGET_SCALE = CKKS_SCALE;

    for (size_t s = 0; s < fd; ++s) {
        Evaluator eval(context);
        Ciphertext row_acc;
        bool init = false;

        for (size_t d = 0; d < DIM; ++d) {
            Ciphertext prod;
            eval.multiply_plain(queryset[d], db[s][d], prod);
            eval.rescale_to_next_inplace(prod);
            prod.scale() = TARGET_SCALE;

            if (!init) {
                row_acc = std::move(prod);
                row_acc.scale() = TARGET_SCALE;
                init = true;
            } else {
                add_aligned_inplace(eval, row_acc, prod, TARGET_SCALE);
            }
        }
        result[s] = std::move(row_acc);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[CKKS-NAIVE] Compute time: " << ms << " ms\n";

    if (data_mode == EmbMode::Test) {
        if (result.empty()) {
            std::cout << "[CKKS-NAIVE TEST] result empty.\n";
            return;
        }

        Plaintext pt;
        decryptor.decrypt(result[0], pt);
        std::vector<double> slots;
        encoder.decode(pt, slots);

        std::cout << "[CKKS-NAIVE TEST] first 16 slots (gid, expected, decrypted):\n";
        size_t print_cnt = std::min<size_t>(16, slots.size());
        for (size_t i = 0; i < print_cnt; ++i) {
            size_t gid      = i;
            double expected = static_cast<double>(DIM);
            double dec_val  = slots[i];

            std::cout << "  gid = " << gid
                      << ", expected = " << expected
                      << ", decrypted = " << dec_val
                      << "\n";
        }
        return;
    }

    std::vector<std::pair<double, size_t>> all_scores;
    all_scores.reserve(NUM_EMBEDDINGS);

    size_t base = 0;
    for (size_t s = 0; s < fd; ++s) {
        if (result[s].is_transparent()) {
            base += slot_count;
            continue;
        }

        Plaintext pt;
        decryptor.decrypt(result[s], pt);
        std::vector<double> slots;
        encoder.decode(pt, slots);

        for (size_t j = 0; j < slot_count; ++j) {
            size_t gid = base + j;
            if (gid >= NUM_EMBEDDINGS) break;
            double score = slots[j];
            all_scores.emplace_back(score, gid);
        }
        base += slot_count;
    }

    std::sort(all_scores.begin(), all_scores.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });

    std::set<size_t> dec_top_k;
    for (size_t i = 0; i < TOP_K && i < all_scores.size(); ++i) {
        dec_top_k.insert(all_scores[i].second);
    }

    size_t intersect_cnt = 0;
    for (auto id : dec_top_k)
        if (true_top_k.count(id)) ++intersect_cnt;

    std::cout << "[CKKS-NAIVE] Top-" << TOP_K
              << " overlap (HE vs Plain): "
              << intersect_cnt << " / " << TOP_K << "\n";
}

/* ------------------- 原始 BSGS（单 embedding / 槽）可选保留 ------------------- */
/*（如果你不再需要旧 BSGS，可以删掉这一部分；这里略过实现） */

/* ------------------- BSGS：复槽打包两条 embedding 的版本 ------------------- */

/*
 * Packed BSGS 编码：在 naive 对角线基础上，“一槽两条向量”。
 *
 * 对第 s 个 block：
 *   - 一个 ciphertext 块最多容纳 emb_per_block = 2*slot_count 条 embedding
 *   - 对槽 i（0 <= i < slot_count）：
 *       gid0 = base + 2*i
 *       gid1 = gid0 + 1
 *     对于对角线 d：
 *       idx = (i + d) % DIM
 *       slots[i] = emb[gid0][idx] + i * emb[gid1][idx]
 *
 * 查询向量 q 一律用“纯实”编码：
 *   Q[i] = q[i % DIM]，虚部=0
 *
 * 乘法：
 *   Q * slots = q * (e0 + i e1) = (q e0) + i (q e1)
 * 维度上累加后：
 *   real(slot[i]) ≈ <q, embedding gid0>
 *   imag(slot[i]) ≈ <q, embedding gid1>
 */

std::vector<std::vector<Plaintext>> encode_database_diagonal_ckks_bsgs_packed(
    const SEALContext &context,
    const std::vector<std::vector<double>> &embeddings,
    size_t slot_count,
    size_t dim,
    double scale,
    size_t baby)
{
    const size_t num_embeddings = embeddings.size();
    const size_t emb_per_block  = 2 * slot_count;
    const size_t fd = (num_embeddings + emb_per_block - 1) / emb_per_block;

    std::vector<std::vector<Plaintext>> db(fd, std::vector<Plaintext>(dim));

    #pragma omp parallel
    {
        CKKSEncoder encoder_local(context);

        #pragma omp for schedule(static)
        for (size_t s = 0; s < fd; ++s) {
            size_t base = s * emb_per_block;

            for (size_t d = 0; d < dim; ++d) {
                std::vector<std::complex<double>> slots(slot_count, {0.0, 0.0});

                // naive 对角线 + “两条 embedding 打包进一个 complex slot”
                for (size_t i = 0; i < slot_count; ++i) {
                    size_t gid0 = base + 2 * i;
                    size_t gid1 = gid0 + 1;

                    double re = 0.0, im = 0.0;
                    if (gid0 < num_embeddings) {
                        size_t idx0 = (i + d) % dim;
                        re = embeddings[gid0][idx0];
                    }
                    if (gid1 < num_embeddings) {
                        size_t idx1 = (i + d) % dim;
                        im = embeddings[gid1][idx1];
                    }
                    slots[i] = std::complex<double>(re, im);
                }

                // BSGS giant-step 预补偿：右移 r = j*baby
                size_t j = d / baby;
                size_t r = (j * baby) % slot_count;
                shift_slots_cyclic(slots, r); // RotRight(P, j*baby)

                encoder_local.encode(slots, scale, db[s][d]);
            }
        }
    }
    return db;
}

/* BSGS baby-step 查询集：Q_k = Rot(Q,k)，k=0..baby-1 */
std::vector<Ciphertext> make_queryset_ckks_bsgs(
    const SEALContext &context,
    const Ciphertext &query,
    const GaloisKeys &galois_keys,
    size_t baby)
{
    std::vector<Ciphertext> qs(baby);
    qs[0] = query;

    #pragma omp parallel
    {
        Evaluator eval_local(context);

        #pragma omp for schedule(static)
        for (size_t k = 1; k < baby; ++k) {
            eval_local.rotate_vector(query, static_cast<int>(k), galois_keys, qs[k]);
        }
    }
    return qs;
}

/* CKKS + BSGS（复槽打包）主流程 */
void vector_match_ckks_bsgs_packed(
    size_t POLY_DEGREE,
    size_t NUM_EMBEDDINGS,
    size_t DIM,
    size_t OMP_THREADS,
    EmbMode data_mode,
    uint32_t data_seed,
    uint32_t query_seed,
    double CKKS_SCALE,
    size_t TOP_K)
{
    omp_set_num_threads(static_cast<int>(OMP_THREADS));

    bool L2_NORMALIZE = true;

    auto embeddings = generate_embeddings_ckks(
        NUM_EMBEDDINGS, DIM,
        data_mode, data_seed, 1.0, 0.9, L2_NORMALIZE);

    std::vector<double> qvec;
    size_t random_idx = 0;

    if (data_mode == EmbMode::Test) {
        qvec.assign(DIM, 1.0);
    } else {
        std::mt19937 gen(query_seed ? query_seed : std::random_device{}());
        std::uniform_int_distribution<size_t> dis(0, NUM_EMBEDDINGS - 1);
        random_idx = dis(gen);
        qvec = embeddings[random_idx];
    }

    std::set<size_t> true_top_k;
    if (data_mode != EmbMode::Test) {
        true_top_k = get_top_k_indices_ckks(qvec, embeddings, TOP_K);
    }

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(POLY_DEGREE);
    parms.set_coeff_modulus(CoeffModulus::Create(
        POLY_DEGREE, { 40, 29, 40 }));

    SEALContext context(parms);
    if (!context.parameters_set()) {
        throw std::runtime_error("[CKKS-BSGS-PACKED] encryption parameters are not valid.");
    }

    KeyGenerator keygen(context);
    PublicKey  pk;  keygen.create_public_key(pk);
    SecretKey  sk = keygen.secret_key();
    GaloisKeys gk;  keygen.create_galois_keys(gk);

    Encryptor   encryptor(context, pk);
    Decryptor   decryptor(context, sk);
    CKKSEncoder encoder(context);

    const size_t slot_count = encoder.slot_count(); // = POLY_DEGREE / 2
    if (DIM > slot_count || (slot_count % DIM) != 0) {
        throw std::invalid_argument(
            "[CKKS-BSGS-PACKED] require DIM <= slot_count and slot_count % DIM == 0.");
    }
    const size_t emb_per_block  = 2 * slot_count;
    const size_t fd = (NUM_EMBEDDINGS + emb_per_block - 1) / emb_per_block;
    const auto [baby, giant] = plan_bsgs(DIM);

    std::cerr << "[CKKS-BSGS-PACKED] poly_degree=" << POLY_DEGREE
              << ", slot_count=" << slot_count
              << ", NUM_EMBEDDINGS=" << NUM_EMBEDDINGS
              << ", DIM=" << DIM
              << ", baby=" << baby << ", giant=" << giant
              << ", fd=" << fd << "\n";

    auto db = encode_database_diagonal_ckks_bsgs_packed(
        context, embeddings, slot_count, DIM, CKKS_SCALE, baby);

    Ciphertext query_ct = encrypt_query_from_vec_ckks(
        encoder, encryptor, slot_count, DIM, qvec, CKKS_SCALE);
    
    auto t0 = std::chrono::high_resolution_clock::now();

    auto queryset = make_queryset_ckks_bsgs(context, query_ct, gk, baby);
    auto t02 = std::chrono::high_resolution_clock::now();

    std::vector<Ciphertext> result(fd);
    const double TARGET_SCALE = CKKS_SCALE;

    for (size_t s = 0; s < fd; ++s) {
        std::vector<Ciphertext> row_part(giant);

        #pragma omp parallel for schedule(static)
        for (size_t j = 0; j < giant; ++j) {
            Evaluator eval_local(context);
            Ciphertext acc_j;
            bool init_local = false;

            for (size_t k = 0; k < baby; ++k) {
                size_t d = j * baby + k;
                if (d >= DIM) break;

                Ciphertext prod;
                eval_local.multiply_plain(queryset[k], db[s][d], prod);
                eval_local.rescale_to_next_inplace(prod);
                prod.scale() = TARGET_SCALE;

                if (!init_local) {
                    acc_j = std::move(prod);
                    acc_j.scale() = TARGET_SCALE;
                    init_local = true;
                } else {
                    add_aligned_inplace(eval_local, acc_j, prod, TARGET_SCALE);
                }
            }

            if (!init_local) continue;

            if (j != 0) {
                eval_local.rotate_vector_inplace(
                    acc_j, static_cast<int>(j * baby), gk);
            }
            row_part[j] = std::move(acc_j);
        }

        // 聚合 giant 部分
        Evaluator eval_row(context);
        Ciphertext row_acc;
        bool init_row = false;

        for (size_t j = 0; j < giant; ++j) {
            if (row_part[j].is_transparent()) continue;
            if (!init_row) {
                row_acc = row_part[j];
                row_acc.scale() = TARGET_SCALE;
                init_row = true;
            } else {
                add_aligned_inplace(eval_row, row_acc, row_part[j], TARGET_SCALE);
            }
        }
        result[s] = std::move(row_acc);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    auto ms0 = std::chrono::duration_cast<std::chrono::milliseconds>(t02 - t0).count();
    auto ms1 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t02).count();
    std::cout << "[CKKS-BSGS-PACKED] Compute time: " << ms << " ms\n";
    std::cout << "[CKKS-BSGS-PACKED] QueryBS: " << ms0 << " ms\n";
    std::cout << "[CKKS-BSGS-PACKED] Multi: " << ms1 << " ms\n";

    if (data_mode == EmbMode::Test) {
        if (result.empty()) {
            std::cout << "[CKKS-BSGS-PACKED TEST] result empty.\n";
            return;
        }

        Plaintext pt;
        decryptor.decrypt(result[0], pt);
        std::vector<std::complex<double>> slots;
        encoder.decode(pt, slots);

        std::cout << "[CKKS-BSGS-PACKED TEST] first 8 slots (gid0,gid1, expected, real, imag):\n";
        size_t print_cnt = std::min<size_t>(8, slots.size());
        for (size_t i = 0; i < print_cnt; ++i) {
            size_t gid0     = 2 * i;
            size_t gid1     = gid0 + 1;
            double expected = static_cast<double>(DIM); // 全1·全1 = DIM

            std::cout << "  slot " << i
                      << " -> gid0=" << gid0
                      << ", gid1="   << gid1
                      << ", expected=" << expected
                      << ", real="   << slots[i].real()
                      << ", imag="   << slots[i].imag()
                      << "\n";
        }
        return;
    }

    // 非 Test：Top-K overlap
    std::vector<std::pair<double, size_t>> all_scores;
    all_scores.reserve(NUM_EMBEDDINGS);

    size_t base_block = 0;
    for (size_t s = 0; s < fd; ++s) {
        if (result[s].is_transparent()) {
            base_block += emb_per_block;
            continue;
        }

        Plaintext pt;
        decryptor.decrypt(result[s], pt);
        std::vector<std::complex<double>> slots;
        encoder.decode(pt, slots);

        for (size_t j = 0; j < slot_count; ++j) {
            size_t gid0 = base_block + 2 * j;
            if (gid0 >= NUM_EMBEDDINGS) break;

            size_t gid1 = gid0 + 1;
            const auto &v = slots[j];

            // real → gid0
            all_scores.emplace_back(v.real(), gid0);
            // imag → gid1（可能溢出 NUM_EMBEDDINGS）
            if (gid1 < NUM_EMBEDDINGS) {
                all_scores.emplace_back(v.imag(), gid1);
            }
        }
        base_block += emb_per_block;
    }

    std::sort(all_scores.begin(), all_scores.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });

    std::set<size_t> dec_top_k;
    for (size_t i = 0; i < TOP_K && i < all_scores.size(); ++i) {
        dec_top_k.insert(all_scores[i].second);
    }

    size_t intersect_cnt = 0;
    for (auto id : dec_top_k)
        if (true_top_k.count(id)) ++intersect_cnt;

    std::cout << "[CKKS-BSGS-PACKED] Top-" << TOP_K
              << " overlap (HE vs Plain): "
              << intersect_cnt << " / " << TOP_K << "\n";
}

/* ------------------- 示例 main：直接跑 PACKED BSGS ------------------- */

int vector_match() {
    size_t POLY_DEGREE     = 4096;
    size_t NUM_EMBEDDINGS  = 4096;
    size_t DIM             = 1024;
    size_t OMP_THREADS     = 64;
    uint32_t DATA_SEED     = 2025;
    uint32_t QUERY_SEED    = 7;
    size_t TOP_K           = 100;
    double CKKS_SCALE      = std::pow(2.0, 30);

    // 1) 可以先用 Test 模式 sanity check
    /*
    vector_match_ckks_bsgs_packed(
        POLY_DEGREE, NUM_EMBEDDINGS, DIM, OMP_THREADS,
        EmbMode::Test, DATA_SEED, QUERY_SEED, CKKS_SCALE, TOP_K);
    */

    // 2) 实际 Uniform 数据 + Cosine
    vector_match_ckks_bsgs_packed(
        POLY_DEGREE, NUM_EMBEDDINGS, DIM, OMP_THREADS,
        EmbMode::Uniform, DATA_SEED, QUERY_SEED, CKKS_SCALE, TOP_K);

    return 0;
}
