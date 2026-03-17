/******************************************************************************
 *  多线程 BGV-BSGS 向量模糊匹配（完整修正版 v4）
 *
 *  面向 SEAL 4.1+ 的关键约定：
 *    1) BGV ciphertext 默认保持在 NTT form
 *    2) rotate_rows / apply_galois 要求 ciphertext 处于 default NTT form
 *    3) add_plain_inplace 要求 ciphertext/plain 都在 default NTT form
 *    4) multiply_plain_inplace 要求 ciphertext/plain 处于相同 NTT form
 *
 *  核心修复：
 *    1) 不再把 freshly encrypted BGV ciphertext 当成“非 NTT”
 *    2) query ciphertext 保持默认 form（即当前 SEAL 版本下的 NTT form）
 *    3) queryset 旋转后不再重复 transform_to_ntt
 *    4) 数据库明文预先转 NTT
 *    5) Euclid/cosine 后处理用到的辅助明文也全部预先转 NTT
 *    6) 多线程改为按 fd（数据库块）并行，避免共享累加状态和异常竞争
 ******************************************************************************/

#include "seal/seal.h"
#include <omp.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>
#include <atomic>
#include <string>

using namespace std;
using namespace seal;

/* ------------------- 小工具结构 ------------------- */
struct BSGSPlan {
    size_t baby;
    size_t giant;
};

static inline BSGSPlan plan_bsgs_bgv(size_t dim) {
    size_t b = static_cast<size_t>(floor(sqrt(static_cast<double>(dim))));
    b = max<size_t>(1, b);
    size_t g = (dim + b - 1) / b;
    return { b, g };
}

static inline size_t ceil_div_bgv(size_t a, size_t b) {
    return (a + b - 1) / b;
}

/* ----------------  数据生成  ---------------- */
enum class EmbMode { Gaussian, Sparse, Uniform, Test };
enum class Metric  { Cosine, Euclid };

vector<vector<double>> generate_embeddings_bgv(
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
            dim_std[d] = 0.5 + 1.5 * frac;
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
        for (size_t i = 0; i < num_embeddings; ++i) {
            double v = static_cast<double>(i + 1);
            fill(emb[i].begin(), emb[i].end(), v);
        }
    } else {
        uniform_real_distribution<double> U(-1.0, 1.0);
        for (size_t i = 0; i < num_embeddings; ++i) {
            for (size_t d = 0; d < dim; ++d) {
                emb[i][d] = U(rng) * scale;
            }
        }
    }

    if (l2_normalize && mode != EmbMode::Test) {
        for (size_t i = 0; i < num_embeddings; ++i) {
            long double norm2 = 0.0L;
            for (size_t d = 0; d < dim; ++d) {
                norm2 += (long double)emb[i][d] * (long double)emb[i][d];
            }
            double norm = sqrt((double)norm2);
            if (norm == 0.0) continue;
            for (size_t d = 0; d < dim; ++d) emb[i][d] /= norm;
        }
    }
    return emb;
}

/* 有符号 <-> Z_t 的严格映射 */
static inline uint64_t map_to_Zt_bgv(long long x, uint64_t t) {
    if (x >= 0) return static_cast<uint64_t>(x) % t;
    uint64_t ax = static_cast<uint64_t>(-x);
    return (t - (ax % t)) % t;
}

static inline long long center_lift_from_Zt_bgv(uint64_t v, uint64_t t) {
    uint64_t half = t >> 1;
    if (v <= half) return static_cast<long long>(v);
    return -static_cast<long long>(t - v);
}

static inline vector<double> compute_l2_norms_bgv(const vector<vector<double>> &E) {
    const double eps = 1e-12;
    vector<double> norms(E.size(), 0.0);
    for (size_t i = 0; i < E.size(); ++i) {
        long double s = 0.0L;
        for (double x : E[i]) s += (long double)x * (long double)x;
        double n = sqrt((double)s);
        norms[i] = (n < eps ? eps : n);
    }
    return norms;
}

static inline long long absmax_scaled_matrix_bgv(
    const vector<vector<double>> &M,
    long long scale_int)
{
    long double m = 0.0L;
    for (const auto &row : M) {
        for (double x : row) {
            m = max(m, fabsl((long double)x * (long double)scale_int));
        }
    }
    long double lim = min<long double>(m, (long double)LLONG_MAX);
    return max<long long>(1LL, (long long)ceil(lim));
}

static inline long long absmax_scaled_vec_bgv(
    const vector<double> &v,
    long long scale_int)
{
    long double m = 0.0L;
    for (double x : v) {
        m = max(m, fabsl((long double)x * (long double)scale_int));
    }
    long double lim = min<long double>(m, (long double)LLONG_MAX);
    return max<long long>(1LL, (long long)ceil(lim));
}

/* ----------------  明文参考结果  ---------------- */
double cosine_similarity_bgv(const vector<double> &vec1, const vector<double> &vec2) {
    double dot_product = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) dot_product += vec1[i] * vec2[i];
    return dot_product;
}

static inline set<size_t> topk_by_euclid_bgv(
    const vector<double>& q,
    const vector<vector<double>>& E,
    size_t k)
{
    vector<pair<double,size_t>> v;
    v.reserve(E.size());
    for (size_t i = 0; i < E.size(); ++i) {
        long double acc = 0.0L;
        for (size_t d = 0; d < q.size(); ++d) {
            long double diff = (long double)q[d] - (long double)E[i][d];
            acc += diff * diff;
        }
        v.emplace_back((double)acc, i);
    }

    if (k < v.size()) {
        nth_element(v.begin(), v.begin() + k, v.end(),
                    [](const auto &a, const auto &b){ return a.first < b.first; });
        v.resize(k);
        sort(v.begin(), v.end(),
             [](const auto &a, const auto &b){ return a.first < b.first; });
    } else {
        sort(v.begin(), v.end(),
             [](const auto &a, const auto &b){ return a.first < b.first; });
    }

    set<size_t> ans;
    for (auto &p : v) ans.insert(p.second);
    return ans;
}

set<size_t> get_top_k_indices_bgv(const vector<double>& query,
                                  const vector<vector<double>>& db,
                                  size_t k) {
    vector<pair<double, size_t>> similarities(db.size());
    for (size_t i = 0; i < db.size(); ++i) {
        similarities[i] = { cosine_similarity_bgv(query, db[i]), i };
    }
    sort(similarities.rbegin(), similarities.rend());

    set<size_t> top_k_indices;
    for (size_t i = 0; i < k && i < similarities.size(); ++i) {
        top_k_indices.insert(similarities[i].second);
    }
    return top_k_indices;
}

/* ----------------  行内循环移位  ---------------- */
static inline void shift_rows_cyclic_half_bgv(vector<uint64_t> &v, size_t sft) {
    size_t n = v.size();
    if (!n) return;
    size_t half = n >> 1;
    sft %= half;
    if (!sft) return;

    vector<uint64_t> tmp(n);
    for (size_t i = 0; i < half; ++i) tmp[(i + sft) % half] = v[i];
    for (size_t i = 0; i < half; ++i) tmp[half + ((i + sft) % half)] = v[half + i];
    v.swap(tmp);
}

/* ----------------  t_bits 估计  ---------------- */
static inline int choose_t_bits_for_euclid_bgv(
    long long A, long long B, size_t dim, int extra_bits = 3)
{
    if (A <= 0) A = 1;
    if (B <= 0) B = 1;
    if (dim == 0) dim = 1;

    __int128 a = (__int128)A;
    __int128 b = (__int128)B;
    __int128 d = (__int128)dim;

    __int128 M = d*a*a + d*b*b + 2*d*a*b;
    __int128 bound = 2 * M;

    int bits = 0;
    __int128 v = bound - 1;
    while (v > 0) { v >>= 1; ++bits; }
    return bits + extra_bits;
}

static inline int choose_t_bits_from_bounds_min_extra_bgv(
    long long A, long long B, size_t dim, long long K, int extra_bits = 3)
{
    if (A <= 0) A = 1;
    if (B <= 0) B = 1;
    if (K <= 0) K = 1;

    __int128 bound = 2;
    bound *= (__int128)dim;
    bound *= (__int128)A;
    bound *= (__int128)B;
    bound *= (__int128)K;

    int bits = 0;
    __int128 v = bound - 1;
    while (v > 0) { v >>= 1; ++bits; }
    return bits + extra_bits;
}

/* ----------------  NTT 明文辅助 ---------------- */
static inline void ensure_plain_ntt_bgv(
    const SEALContext &context,
    Plaintext &pt)
{
    Evaluator eval(context);
    if (!pt.is_ntt_form()) {
        eval.transform_to_ntt_inplace(pt, context.first_parms_id());
    }
    if (!pt.is_ntt_form()) {
        throw logic_error("failed to transform plaintext to NTT form");
    }
}

/* ----------------  辅助明文打包  ---------------- */
static inline vector<Plaintext> pack_inv_norms_per_row_bgv(
    const SEALContext &context,
    const vector<double> &norms,
    size_t row_size,
    uint64_t plain_modulus_t,
    long long inv_scale_int
) {
    const size_t num_embeddings = norms.size();
    const size_t fd = (num_embeddings + row_size - 1) / row_size;
    vector<Plaintext> row_pts(fd);

    #pragma omp parallel
    {
        BatchEncoder enc_local(context);
        Evaluator eval_local(context);
        const size_t slot_count = enc_local.slot_count();

        #pragma omp for schedule(static)
        for (size_t s = 0; s < fd; ++s) {
            vector<uint64_t> slots(slot_count, 0ULL);
            for (size_t i = 0; i < row_size; ++i) {
                size_t gid = s * row_size + i;
                if (gid >= num_embeddings) break;

                long double invn = 1.0L / (long double)norms[gid];
                long long q = llround(invn * (long double)inv_scale_int);
                slots[i] = map_to_Zt_bgv(q, plain_modulus_t);
            }
            enc_local.encode(slots, row_pts[s]);
            eval_local.transform_to_ntt_inplace(row_pts[s], context.first_parms_id());
        }
    }
    return row_pts;
}

static inline vector<Plaintext> pack_bnorm2_per_row_bgv(
    const SEALContext &context,
    const vector<vector<double>> &embeddings,
    size_t row_size, size_t dim,
    uint64_t plain_modulus_t,
    long long scale_int)
{
    const size_t num_embeddings = embeddings.size();
    const size_t fd = (num_embeddings + row_size - 1) / row_size;
    vector<Plaintext> row_pts(fd);

    #pragma omp parallel
    {
        BatchEncoder enc_local(context);
        Evaluator eval_local(context);
        const size_t slot_count = enc_local.slot_count();

        #pragma omp for schedule(static)
        for (size_t s = 0; s < fd; ++s) {
            vector<uint64_t> slots(slot_count, 0ULL);

            for (size_t i = 0; i < row_size; ++i) {
                size_t gid = s * row_size + i;
                if (gid >= num_embeddings) break;

                __int128 acc = 0;
                for (size_t d = 0; d < dim; ++d) {
                    long double v = (long double)embeddings[gid][d] * (long double)scale_int;
                    long long qs = llround(v);
                    acc += (__int128)qs * (__int128)qs;
                }
                long long b2 = (long long)std::min<__int128>(acc, LLONG_MAX);
                slots[i] = map_to_Zt_bgv(b2, plain_modulus_t);
            }

            enc_local.encode(slots, row_pts[s]);
            eval_local.transform_to_ntt_inplace(row_pts[s], context.first_parms_id());
        }
    }
    return row_pts;
}

/* ----------------  数据库对角线编码  ---------------- */
vector<vector<Plaintext>> encode_database_diagonal_parallel_bgv(
        const SEALContext &context,
        const vector<vector<double>> &emb,
        size_t row_size, size_t dim,
        long long scale_int,
        uint64_t plain_modulus_t,
        int db_coeff)
{
    const size_t num_embeddings = emb.size();
    const size_t fd = (num_embeddings + row_size - 1) / row_size;
    const auto [baby, giant] = plan_bsgs_bgv(dim);
    (void)giant;

    vector<vector<Plaintext>> db(fd, vector<Plaintext>(dim));

    #pragma omp parallel
    {
        BatchEncoder enc_local(context);
        Evaluator eval_local(context);
        const size_t slot_count = enc_local.slot_count();

        #pragma omp for schedule(static)
        for (size_t s = 0; s < fd; ++s) {
            for (size_t d = 0; d < dim; ++d) {
                vector<uint64_t> msg(slot_count, 0ULL);

                for (size_t i = 0; i < row_size; ++i) {
                    size_t row_idx = s * row_size + i;
                    if (row_idx >= num_embeddings) break;

                    double ele = emb[row_idx][(i + d) % dim];
                    long long q = llround((long double)ele * (long double)scale_int);
                    q *= (long long)db_coeff;
                    msg[i] = map_to_Zt_bgv(q, plain_modulus_t);
                }

                size_t j = d / baby;
                size_t r = (j * baby) % row_size;
                shift_rows_cyclic_half_bgv(msg, r);

                enc_local.encode(msg, db[s][d]);
                eval_local.transform_to_ntt_inplace(db[s][d], context.first_parms_id());
            }
        }
    }

    return db;
}

/* ----------------  查询向量加密  ---------------- */
Ciphertext encrypt_query_from_vec_bgv(
    BatchEncoder &enc, Encryptor &encor,
    size_t row_size, size_t dim,
    const vector<double> &qvecF,
    long long scale_int,
    uint64_t plain_modulus_t)
{
    const size_t slot_count = enc.slot_count();
    vector<uint64_t> slots(slot_count, 0ULL);

    for (size_t i = 0; i < row_size; ++i) {
        double ele = qvecF[i % dim];
        long double scaled = (long double)ele * (long double)scale_int;
        if (scaled > (long double)numeric_limits<long long>::max() ||
            scaled < (long double)numeric_limits<long long>::min()) {
            throw overflow_error("scaled value out of int64 range");
        }
        long long q = llround(scaled);
        slots[i] = map_to_Zt_bgv(q, plain_modulus_t);
    }

    for (size_t i = row_size; i < slot_count; ++i) {
        slots[i] = 0ULL;
    }

    Plaintext pt;
    enc.encode(slots, pt);

    Ciphertext ct;
    encor.encrypt(pt, ct);
    return ct;
}

/* 输入 query 直接按当前 SEAL 默认 form 使用（BGV: default NTT form） */
vector<Ciphertext> make_queryset_bgv(
    const SEALContext &context,
    const Ciphertext  &query_ct,
    const GaloisKeys  &gk,
    size_t baby)
{
    vector<Ciphertext> qs(baby);
    qs[0] = query_ct;

    #pragma omp parallel
    {
        Evaluator eval(context);
        #pragma omp for schedule(static)
        for (size_t d = 1; d < baby; ++d) {
            eval.rotate_rows(query_ct, static_cast<int>(d), gk, qs[d]);
        }
    }

    return qs;
}

/* 单个 fd 行计算：整个过程中保持 default NTT form */
Ciphertext compute_one_row_bsgs_bgv(
    const SEALContext &context,
    const vector<Ciphertext> &queryset,
    const vector<Plaintext> &db_row,
    const GaloisKeys &gk,
    size_t baby,
    size_t giant,
    size_t dim,
    size_t row_size)
{
    Evaluator eval(context);

    Ciphertext row_acc;
    bool row_init = false;

    for (size_t j = 0; j < giant; ++j) {
        Ciphertext giant_acc;
        bool giant_init = false;

        for (size_t k = 0; k < baby; ++k) {
            size_t d = j * baby + k;
            if (d >= dim) break;

            if (!queryset[k].is_ntt_form()) {
                throw logic_error("BGV queryset ciphertext must be in NTT form");
            }
            if (!db_row[d].is_ntt_form()) {
                throw logic_error("database plaintext must be in NTT form");
            }

            Ciphertext prod;
            eval.multiply_plain(queryset[k], db_row[d], prod);

            if (!prod.is_ntt_form()) {
                throw logic_error("multiply_plain result is expected to stay in NTT form");
            }

            if (!giant_init) {
                giant_acc = std::move(prod);
                giant_init = true;
            } else {
                eval.add_inplace(giant_acc, prod);
            }
        }

        if (!giant_init) continue;

        if (j) {
            size_t rot = j * baby;
            if (rot >= row_size) {
                throw logic_error("rotate step exceeds row size");
            }
            if (!giant_acc.is_ntt_form()) {
                throw logic_error("BGV giant_acc must be in default NTT form before rotate_rows");
            }
            eval.rotate_rows_inplace(giant_acc, static_cast<int>(rot), gk);
        }

        if (!row_init) {
            row_acc = std::move(giant_acc);
            row_init = true;
        } else {
            eval.add_inplace(row_acc, giant_acc);
        }
    }

    return row_acc;
}

/* -------------------  主流程（BGV） ------------------- */
void vector_match_bgv(
    size_t POLY_DEGREE,
    size_t NUM_EMBEDDINGS,
    size_t DIM,
    size_t OMP_THREADS,
    size_t PlainModulusBits,
    EmbMode data_mode,
    EmbMode query_mode,
    uint32_t data_seed,
    uint32_t query_seed,
    long long SCALE_INT,
    size_t TOP_K,
    bool L2_NORMALIZE,
    Metric metric = Metric::Cosine
){
    omp_set_num_threads(static_cast<int>(OMP_THREADS));

    auto embeddings = generate_embeddings_bgv(
        NUM_EMBEDDINGS, DIM, data_mode, data_seed, 1.0, 0.9, L2_NORMALIZE);

    vector<double> qvecF;
    size_t random_idx = 0;

    if (query_mode == EmbMode::Test || data_mode == EmbMode::Test) {
        qvecF.assign(DIM, 1.0);
    } else if (query_mode == data_mode) {
        std::mt19937 gen(query_seed ? query_seed : std::random_device{}());
        std::uniform_int_distribution<size_t> dis(0, NUM_EMBEDDINGS - 1);
        random_idx = dis(gen);
        qvecF = embeddings[random_idx];
    } else {
        auto qtmp = generate_embeddings_bgv(
            1, DIM, query_mode, query_seed, 1.0, 0.9, L2_NORMALIZE);
        qvecF = std::move(qtmp[0]);
    }

    set<size_t> true_top_k;
    vector<double> b_norms_for_inv;

    if (data_mode != EmbMode::Test) {
        if (metric == Metric::Euclid) {
            true_top_k = topk_by_euclid_bgv(qvecF, embeddings, TOP_K);
        } else if (L2_NORMALIZE) {
            true_top_k = get_top_k_indices_bgv(qvecF, embeddings, TOP_K);
        } else {
            b_norms_for_inv = compute_l2_norms_bgv(embeddings);
            vector<pair<double,size_t>> sims;
            sims.reserve(NUM_EMBEDDINGS);

            for (size_t i = 0; i < NUM_EMBEDDINGS; ++i) {
                long double dot = 0.0L;
                for (size_t d = 0; d < DIM; ++d) {
                    dot += (long double)qvecF[d] * (long double)embeddings[i][d];
                }
                long double score = dot / (long double)b_norms_for_inv[i];
                sims.emplace_back((double)score, i);
            }

            sort(sims.begin(), sims.end(),
                 [](const auto &a, const auto &b){ return a.first > b.first; });
            for (size_t k = 0; k < TOP_K && k < sims.size(); ++k) {
                true_top_k.insert(sims[k].second);
            }
        }
    }

    long long A = absmax_scaled_matrix_bgv(embeddings, SCALE_INT);
    long long B = absmax_scaled_vec_bgv(qvecF, SCALE_INT);

    long long K = 1;
    int t_bits = 0;

    if (metric == Metric::Cosine) {
        if (!L2_NORMALIZE) {
            if (b_norms_for_inv.empty()) b_norms_for_inv = compute_l2_norms_bgv(embeddings);
            long double inv_max = 0.0L;
            for (double nb : b_norms_for_inv) {
                inv_max = max(inv_max, 1.0L / (long double)nb);
            }
            long long inv_scale_int = SCALE_INT;
            long long K_tmp = (long long)ceil((long double)inv_scale_int * inv_max);
            if (K_tmp < 1) K_tmp = 1;
            K = K_tmp;
        }
        t_bits = choose_t_bits_from_bounds_min_extra_bgv(A, B, DIM, K, 3);
    } else {
        t_bits = choose_t_bits_for_euclid_bgv(A, B, DIM, 3);
    }

    cerr << "[info][BGV] estimated t_bits = " << t_bits
         << " (A=" << A << ", B=" << B << ", D=" << DIM << ", K=" << K << ")\n";

    if (data_mode != EmbMode::Test) {
        PlainModulusBits = max<size_t>(PlainModulusBits, (size_t)t_bits);
    }

    if (PlainModulusBits < 2) PlainModulusBits = 2;
    if (PlainModulusBits > 50) {
        cerr << "[warn][BGV] PlainModulusBits too large for safe batching; clamp to 50.\n";
        PlainModulusBits = 50;
    }

    EncryptionParameters parms(scheme_type::bgv);
    parms.set_poly_modulus_degree(POLY_DEGREE);
    parms.set_coeff_modulus(CoeffModulus::BFVDefault(POLY_DEGREE));
    parms.set_plain_modulus(PlainModulus::Batching(POLY_DEGREE, (int)PlainModulusBits));

    cout << PlainModulusBits << " bits for plaintext modulus.\n";

    SEALContext context(parms);
    if (!context.parameters_set()) {
        throw runtime_error("[BGV] encryption parameters are not valid.");
    }

    KeyGenerator keygen(context);
    PublicKey pk;
    keygen.create_public_key(pk);
    SecretKey sk = keygen.secret_key();
    GaloisKeys gk;
    keygen.create_galois_keys(gk);

    Encryptor encryptor(context, pk);
    Decryptor decryptor(context, sk);
    BatchEncoder encoder(context);

    const size_t degreeN = parms.poly_modulus_degree();
    assert(degreeN == POLY_DEGREE && "degree mismatch");

    const size_t slot_count = encoder.slot_count();
    const size_t row_size   = slot_count / 2;

    const uint64_t plain_modulus_t =
        context.first_context_data()->parms().plain_modulus().value();

    Plaintext qnorm2_pt;
    vector<Plaintext> bnorm2_row_pts;

    if (metric == Metric::Euclid) {
        __int128 acc_q = 0;
        for (size_t d = 0; d < DIM; ++d) {
            long long qs = llround((long double)qvecF[d] * (long double)SCALE_INT);
            acc_q += (__int128)qs * (__int128)qs;
        }
        long long q2 = (long long)min<__int128>(acc_q, LLONG_MAX);

        vector<uint64_t> slots(slot_count, 0ULL);
        for (size_t i = 0; i < row_size; ++i) {
            slots[i] = map_to_Zt_bgv(q2, plain_modulus_t);
        }
        encoder.encode(slots, qnorm2_pt);
        ensure_plain_ntt_bgv(context, qnorm2_pt);

        bnorm2_row_pts = pack_bnorm2_per_row_bgv(
            context, embeddings, row_size, DIM, plain_modulus_t, SCALE_INT);
    }

    vector<Plaintext> invnorm_row_pts;
    if (metric == Metric::Cosine && !L2_NORMALIZE) {
        if (b_norms_for_inv.empty()) b_norms_for_inv = compute_l2_norms_bgv(embeddings);
        long long inv_scale_int = SCALE_INT;
        invnorm_row_pts = pack_inv_norms_per_row_bgv(
            context, b_norms_for_inv, row_size, plain_modulus_t, inv_scale_int);
    }

    const auto [baby, giant] = plan_bsgs_bgv(DIM);
    const size_t fd = ceil_div_bgv(NUM_EMBEDDINGS, row_size);

    int db_coeff = (metric == Metric::Euclid ? -2 : +1);
    auto db = encode_database_diagonal_parallel_bgv(
        context, embeddings, row_size, DIM, SCALE_INT, plain_modulus_t, db_coeff);

    Ciphertext query_ct = encrypt_query_from_vec_bgv(
        encoder, encryptor, row_size, DIM, qvecF, SCALE_INT, plain_modulus_t);

    if (!query_ct.is_ntt_form()) {
        throw logic_error("BGV fresh ciphertext is expected to be in NTT form in this SEAL version");
    }

    auto t0 = chrono::high_resolution_clock::now();

    auto queryset = make_queryset_bgv(context, query_ct, gk, baby);
    for (size_t i = 0; i < queryset.size(); ++i) {
        if (!queryset[i].is_ntt_form()) {
            throw logic_error("BGV rotated query ciphertext is expected to remain in NTT form");
        }
    }

    vector<Ciphertext> result(fd);
    atomic<bool> failed(false);
    string first_error;

    #pragma omp parallel
    {
        Evaluator eval(context);

        #pragma omp for schedule(static)
        for (long long i_ll = 0; i_ll < (long long)fd; ++i_ll) {
            size_t i = (size_t)i_ll;

            try {
                Ciphertext row_acc = compute_one_row_bsgs_bgv(
                    context, queryset, db[i], gk, baby, giant, DIM, row_size);

                if (row_acc.parms_id() != parms_id_zero) {
                    if (!row_acc.is_ntt_form()) {
                        throw logic_error("BGV row_acc must remain in NTT form before post-processing");
                    }

                    if (metric == Metric::Cosine) {
                        if (!L2_NORMALIZE) {
                            if (!invnorm_row_pts[i].is_ntt_form()) {
                                throw logic_error("invnorm plaintext must be in NTT form");
                            }
                            eval.multiply_plain_inplace(row_acc, invnorm_row_pts[i]);
                        }
                    } else {
                        if (!qnorm2_pt.is_ntt_form() || !bnorm2_row_pts[i].is_ntt_form()) {
                            throw logic_error("Euclid auxiliary plaintexts must be in NTT form");
                        }
                        eval.add_plain_inplace(row_acc, qnorm2_pt);
                        eval.add_plain_inplace(row_acc, bnorm2_row_pts[i]);
                    }
                }

                result[i] = std::move(row_acc);
            } catch (const std::exception &e) {
                #pragma omp critical
                {
                    if (!failed.load()) {
                        failed.store(true);
                        first_error = e.what();
                    }
                }
            } catch (...) {
                #pragma omp critical
                {
                    if (!failed.load()) {
                        failed.store(true);
                        first_error = "unknown exception in parallel row computation";
                    }
                }
            }
        }
    }

    if (failed.load()) {
        throw runtime_error(first_error);
    }

    auto t1 = chrono::high_resolution_clock::now();
    auto ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
    cout << "[BGV] Compute time: " << ms << " ms\n";

    if (data_mode == EmbMode::Test) {
        Plaintext res_pt;
        decryptor.decrypt(result[0], res_pt);

        vector<uint64_t> u;
        encoder.decode(res_pt, u);

        const long double scale_sq =
            (long double)SCALE_INT * (long double)SCALE_INT;

        cout << "[BGV TEST] first 16: ";
        for (size_t i = 0; i < 16 && i < row_size; ++i) {
            long long s = center_lift_from_Zt_bgv(u[i], plain_modulus_t);
            long double real_v = (long double)s / scale_sq;
            cout << (double)real_v << (i + 1 < 16 ? ", " : "\n");
        }
    } else {
        vector<pair<double, size_t>> all_scores;
        all_scores.reserve(NUM_EMBEDDINGS);

        const long double post_div =
            (metric == Metric::Cosine)
            ? ( L2_NORMALIZE
                ? ( (long double)SCALE_INT * SCALE_INT )
                : ( (long double)SCALE_INT * SCALE_INT * (long double)SCALE_INT ) )
            : ( (long double)SCALE_INT * SCALE_INT );

        size_t global_base = 0;

        for (size_t i = 0; i < fd; ++i) {
            if (result[i].parms_id() == parms_id_zero) {
                global_base += row_size;
                continue;
            }

            Plaintext pt;
            decryptor.decrypt(result[i], pt);

            vector<uint64_t> slots_u64;
            encoder.decode(pt, slots_u64);

            for (size_t j = 0; j < row_size; ++j) {
                size_t gid = global_base + j;
                if (gid >= NUM_EMBEDDINGS) break;

                long long signed_v = center_lift_from_Zt_bgv(slots_u64[j], plain_modulus_t);
                long double real_v = (long double)signed_v / post_div;
                if (metric == Metric::Euclid) real_v = -real_v;

                all_scores.emplace_back((double)real_v, gid);
            }

            global_base += row_size;
        }

        sort(all_scores.begin(), all_scores.end(),
             [](const auto &a, const auto &b){ return a.first > b.first; });

        set<size_t> dec_top_k;
        for (size_t i = 0; i < TOP_K && i < all_scores.size(); ++i) {
            dec_top_k.insert(all_scores[i].second);
        }

        size_t intersect_cnt = 0;
        for (auto id : dec_top_k) {
            if (true_top_k.count(id)) ++intersect_cnt;
        }

        cout << "[BGV] Top-" << TOP_K << " overlap (HE vs Plain): "
             << intersect_cnt << " / " << TOP_K << "\n";

        cout << "[BGV] Top-10 decrypted ids: ";
        for (size_t i = 0; i < min<size_t>(10, all_scores.size()); ++i) {
            cout << all_scores[i].second
                 << (i + 1 < min<size_t>(10, all_scores.size()) ? ", " : "\n");
        }
    }
}

/* ------------------- 示例 demo（BGV） ------------------- */
int main2_bgv() {
    vector_match_bgv(4096, 4096, 1024, 64, 20,
                     EmbMode::Uniform, EmbMode::Uniform,
                     2025, 7, 1024, 100, true, Metric::Cosine);
    vector_match_bgv(4096, 4096, 1024, 64, 20,
        EmbMode::Uniform, EmbMode::Uniform,
        2025, 7, 512, 100, true, Metric::Cosine);
    vector_match_bgv(4096, 4096, 1024, 64, 20,
        EmbMode::Uniform, EmbMode::Uniform,
        2025, 7, 256, 100, true, Metric::Cosine);
    vector_match_bgv(4096, 4096, 1024, 64, 20,
        EmbMode::Uniform, EmbMode::Uniform,
        2025, 7, 128, 100, true, Metric::Cosine);
    vector_match_bgv(4096, 4096, 1024, 64, 20,
        EmbMode::Uniform, EmbMode::Uniform,
        2025, 7, 64, 100, true, Metric::Cosine);
    return 0;
}

int example_bfv_basics()
{
    try {
        return main2_bgv();
    } catch (const std::exception &e) {
        std::cerr << "[FATAL] " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "[FATAL] unknown exception" << std::endl;
        return 1;
    }
}