/******************************************************************************
 *  多线程 BFV-BSGS 向量模糊匹配（参数化 & 更真实的embedding）
 *  - poly_modulus_degree   = 可配（默认 4096）
 *  - NUM_EMBEDDINGS        = 可配（默认 4096）
 *  - DIM                   = 1024（按测试要求固定）
 *  - BSGS 自动选择 baby, giant（不要求 sqrt(DIM) 为整数）
 *  - OpenMP 并行
 *  - 无 magic number：循环均由维度/度数自动推导
 *  written by Zehao Chen (refactored + fixed)
 ******************************************************************************/
#include "seal/seal.h"
#include <omp.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
#include <cassert>
#include <algorithm>
#include <limits>
#include <climits>
#include <math.h>
#include <set>

using namespace std;
using namespace seal;

/* ------------------- 用户可调参数 ------------------- */
// constexpr size_t NUM_EMBEDDINGS_DEFAULT = 4096;
// constexpr size_t DIM                    = 1024;     
// constexpr size_t POLY_DEGREE_DEFAULT    = 4096;
// constexpr size_t OMP_THREADS_DEFAULT    = 64;

// 固定小数放大：默认 1000（你也可以根据数据分布调大/调小）
// constexpr long long DEFAULT_SCALE_INT   = 1000LL;

/* ------------------- 小工具函数 ------------------- */
struct BSGSPlan {
    size_t baby;
    size_t giant;
};

// 计算 BSGS 的 baby-step 和 giant-step 大小
static inline BSGSPlan plan_bsgs(size_t dim) {
    size_t b = static_cast<size_t>(floor(sqrt(static_cast<double>(dim))));
    b = max<size_t>(1, b);
    size_t g = (dim + b - 1) / b;
    return { b, g };
}

// 向上取整除法
static inline size_t ceil_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}

/* ----------------  更贴近真实的 embedding 生成  ---------------- */
enum class EmbMode { Gaussian, Sparse, Uniform, Test };
enum class Metric { Cosine, Euclid };

vector<vector<double>> generate_embeddings(
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
        // 第 i 条向量全为 (i+1)，不做归一化
        for (size_t i = 0; i < num_embeddings; ++i) {
            double v = static_cast<double>((i + 1));
            std::fill(emb[i].begin(), emb[i].end(), v);
        }
    } else { // Uniform
        uniform_real_distribution<double> U(-1.0, 1.0);
        for (size_t i = 0; i < num_embeddings; ++i) {
            for (size_t d = 0; d < dim; ++d) {
                emb[i][d] = U(rng) * scale;
            }
        }
    }

    // 归一化每个 embedding 向量的平方和为 1（Test 模式不归一化）
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

/* 有符号 <-> Z_t 的严格映射（中心化）*/
static inline uint64_t map_to_Zt(long long x, uint64_t t) {
    if (x >= 0) return static_cast<uint64_t>(x) % t;
    uint64_t ax = static_cast<uint64_t>(-x);
    return (t - (ax % t)) % t;
}

// 计算每个向量的 L2 范数
static inline vector<double> compute_l2_norms(const vector<vector<double>> &E) {
    const double eps = 1e-12;
    vector<double> norms(E.size(), 0.0);
    for (size_t i = 0; i < E.size(); ++i) {
        long double s = 0.0L;
        for (double x : E[i]) s += (long double)x * (long double)x;
        double n = sqrt((double)s);
        norms[i] = (n < eps ? eps : n); // clip，防止倒数爆炸
    }
    return norms;
}

// 将 1/||b_i|| 量化并按 “每行 degreeN 槽” 打包成明文（普通域）
static inline vector<Plaintext> pack_inv_norms_per_row(
    const SEALContext &context,
    const vector<double> &norms,      // ||b_i||
    size_t degreeN,
    uint64_t plain_modulus_t,
    long long inv_scale_int           // 倒数的量化系数，建议先用 SCALE_INT
) {
    const size_t num_embeddings = norms.size();
    const size_t fd = (num_embeddings + degreeN - 1) / degreeN;

    vector<Plaintext> row_pts(fd);
    BatchEncoder enc(context);

    for (size_t s = 0; s < fd; ++s) {
        vector<uint64_t> slots(degreeN, 0ULL);
        for (size_t i = 0; i < degreeN; ++i) {
            size_t gid = s * degreeN + i;
            if (gid >= num_embeddings) break;
            long double invn = 1.0L / (long double)norms[gid];
            long long q = llround(invn * (long double)inv_scale_int);
            slots[i] = map_to_Zt(q, plain_modulus_t);
        }
        enc.encode(slots, row_pts[s]);  // 普通域
    }
    return row_pts;
}

// 按行打包每条库向量的范数平方（量化后）到明文：
// 返回长度为 fd=ceil(num_embeddings/degreeN) 的明文数组；第 s 个明文的第 i 槽对应 gid=s*degreeN+i。
// 注意：这里编码在普通域（非 NTT），便于后续 add_plain_inplace。
static inline std::vector<seal::Plaintext> pack_bnorm2_per_row(
    const seal::SEALContext &context,
    const std::vector<std::vector<double>> &embeddings,
    size_t degreeN, size_t dim,
    uint64_t plain_modulus_t,
    long long scale_int)
{
    using namespace seal;
    const size_t num_embeddings = embeddings.size();
    const size_t fd = (num_embeddings + degreeN - 1) / degreeN;

    std::vector<Plaintext> row_pts(fd);

    #pragma omp parallel
    {
        BatchEncoder enc_local(context); // ← 线程局部 encoder

        #pragma omp for schedule(static)
        for (size_t s = 0; s < fd; ++s) {
            std::vector<uint64_t> slots(degreeN, 0ULL);

            for (size_t i = 0; i < degreeN; ++i) {
                const size_t gid = s * degreeN + i;
                if (gid >= num_embeddings) break;

                __int128 acc = 0;
                for (size_t d = 0; d < dim; ++d) {
                    long double v = (long double)embeddings[gid][d] * (long double)scale_int;
                    long long qs = llround(v);
                    acc += (__int128)qs * (__int128)qs;
                }
                long long b2 = (long long)std::min<__int128>(acc, LLONG_MAX);
                slots[i] = map_to_Zt(b2, plain_modulus_t);
            }
            enc_local.encode(slots, row_pts[s]); // 普通域
        }
    }
    return row_pts;
}


/* 计算明文余弦相似度（已做归一化时，直接点积）*/
double cosine_similarity(const vector<double> &vec1, const vector<double> &vec2) {
    double dot_product = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) dot_product += vec1[i] * vec2[i];
    return dot_product;
}

static inline set<size_t> topk_by_euclid(
    const vector<double>& q,
    const vector<vector<double>>& E,
    size_t k)
{
    vector<pair<double,size_t>> v; v.reserve(E.size());
    for (size_t i = 0; i < E.size(); ++i) {
        long double acc = 0.0L;
        for (size_t d = 0; d < q.size(); ++d) {
            long double diff = (long double)q[d] - (long double)E[i][d];
            acc += diff * diff;
        }
        v.emplace_back((double)acc, i); // 距离平方
    }
    // 取前 k 小（升序）
    if (k < v.size()) {
        nth_element(v.begin(), v.begin()+k, v.end(),
                    [](auto &a, auto &b){ return a.first < b.first; });
        v.resize(k);
    } else {
        sort(v.begin(), v.end(), [](auto&a, auto&b){ return a.first < b.first; });
    }
    set<size_t> ans;
    for (auto &p : v) ans.insert(p.second);
    return ans;
}


// 取 top-k 最大相似度的索引集合
set<size_t> get_top_k_indices(const vector<double>& query, const vector<vector<double>>& db, size_t k) {
    vector<pair<double, size_t>> similarities(db.size());
    for (size_t i = 0; i < db.size(); ++i) {
        similarities[i] = { cosine_similarity(query, db[i]), i };
    }
    sort(similarities.rbegin(), similarities.rend()); // 按照相似度降序排列

    set<size_t> top_k_indices;
    for (size_t i = 0; i < k && i < similarities.size(); ++i) {
        top_k_indices.insert(similarities[i].second);
    }
    return top_k_indices;
}

// 按行（每行 N/2 槽）右移 sft 步
static inline void shift_rows_cyclic(vector<uint64_t> &vec, size_t shift) {
    size_t n = vec.size();
    vector<uint64_t> temp(n);
    for (size_t i = 0; i < n; ++i) {
        temp[i] = vec[(i + shift) % n];
    }
    vec = std::move(temp);
}

// 按行（每行 N/2 槽）右移 sft 步
static inline void shift_rows_cyclic_half(vector<uint64_t> &v, size_t sft) {
    size_t n = v.size();
    if (!n) return;
    size_t half = n >> 1;
    sft %= half;
    if (!sft) return;
    vector<uint64_t> tmp(n);
    for (size_t i = 0; i < half; ++i) tmp[(i + sft) % half] = v[i];               // 第一行
    for (size_t i = 0; i < half; ++i) tmp[half + ((i + sft) % half)] = v[half+i]; // 第二行
    v.swap(tmp);
}

static inline long long center_lift_from_Zt(uint64_t v, uint64_t t) {
    uint64_t half = t >> 1; // floor(t/2)
    if (v <= half) return static_cast<long long>(v);
    return -static_cast<long long>(t - v);
}

/* 估算所需模数位数：确保内积不会在模 t 上环绕（未启用的参考实现）*/
static inline int choose_t_bits_for_euclid(
    long long A, long long B, size_t dim, int extra_bits = 3)
{
    if (A <= 0) A = 1; if (B <= 0) B = 1; if (dim == 0) dim = 1;
    __int128 a = (__int128)A, b = (__int128)B, d = (__int128)dim;
    __int128 M = d*a*a + d*b*b + 2*d*a*b;      // |q|^2 + |b|^2 - 2<q,b> 的最坏幅度
    __int128 bound = 2 * M;                    // 使 t > 2*M
    int bits = 0; __int128 v = bound - 1; while (v > 0) { v >>= 1; ++bits; }
    return bits + extra_bits;
}


// 返回：ceil(log2( 2*dim*A*B*K )) + extra_bits
// 其中 K = ceil(inv_scale_int * max_i (1/||b_i||)), 若 L2_NORMALIZE=true 则 K=1
static inline int choose_t_bits_from_bounds_min_extra(
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
    __int128 v = bound - 1;   // ceil(log2(bound))
    while (v > 0) { v >>= 1; ++bits; }
    return bits + extra_bits;
}


/* ----------------  数据库对角线+BSGS 编码  ---------------- */
// 改函数签名：传 context 进来，在线程内构造 encoder
vector<vector<Plaintext>> encode_database_diagonal_parallel(
        const SEALContext &context,
        const vector<vector<double>> &emb,
        size_t degreeN, size_t dim,
        long long scale_int,
        uint64_t plain_modulus_t,
        int db_coeff)
{
    const size_t num_embeddings = emb.size();
    const size_t fd = (num_embeddings + degreeN - 1) / degreeN;
    const auto [baby, giant] = plan_bsgs(dim);

    vector<vector<Plaintext>> db(fd, vector<Plaintext>(dim));

    #pragma omp parallel
    {
        // ⚠️ 每个线程各自一个 encoder（线程局部内存池）
        BatchEncoder enc_local(context);

        #pragma omp for schedule(static)
        for (size_t s = 0; s < fd; ++s) {
            for (size_t d = 0; d < dim; ++d) {
                vector<uint64_t> msg(degreeN, 0ULL);

                for (size_t i = 0; i < degreeN; ++i) {
                    size_t row_idx = s * degreeN + i;
                    if (row_idx >= num_embeddings) break;

                    double ele = emb[row_idx][(i + d) % dim];
                    long long q = llround((long double)ele * (long double)scale_int);
                    q *= (long long)db_coeff; // Euclid 非归一化时需要乘以 db_coeff
                    msg[i] = map_to_Zt(q, plain_modulus_t);
                }

                // 你现在用的整段 N 右移在 TEST 下无影响，保留也可以
                // size_t j = d / baby;
                // size_t sft = (degreeN - (j * baby) % degreeN) % degreeN;
                // shift_rows_cyclic(msg, sft);

                size_t j = d / baby;               // giant 索引
                size_t half = degreeN >> 1;
                size_t r = (j * baby) % half;      // 运行期 rotate_rows 的“左移 r”
                size_t sft = r;                    // ✓ 编码端做“右移 r”抵消它
                shift_rows_cyclic_half(msg, sft);       // ✓ 按行右移 r

                enc_local.encode(msg, db[s][d]); // ✅ 线程私有 encoder
            }
        }
    }
    return db;
}


/* ----------------  查询向量生成与加密  ---------------- */
Ciphertext encrypt_query_from_vec(BatchEncoder &enc, Encryptor &encor,
                                  size_t degreeN, size_t dim,
                                  const vector<double> &qvecF,
                                  long long scale_int,
                                  uint64_t plain_modulus_t)
{
    vector<uint64_t> slots(degreeN, 0ULL);
    for (size_t i = 0; i < degreeN; ++i) {
        double ele = qvecF[i % dim];
        long double scaled = static_cast<long double>(ele) * static_cast<long double>(scale_int);
        if (scaled > static_cast<long double>(std::numeric_limits<long long>::max()) ||
            scaled < static_cast<long double>(std::numeric_limits<long long>::min())) {
            throw std::overflow_error("scaled value out of int64 range");
        }
        long long q = llround(scaled);
        slots[i] = map_to_Zt(q, plain_modulus_t);
    }

    Plaintext pt;
    enc.encode(slots, pt);
    Ciphertext ct;
    encor.encrypt(pt, ct);
    return ct;
}

/* 生成 baby-step 旋转的 ciphertext 数组 */
vector<Ciphertext> make_queryset(const SEALContext &context,
                                 const Ciphertext  &query,
                                 const GaloisKeys  &gk,
                                 size_t baby)
{
    vector<Ciphertext> qs(baby);
    qs[0] = query;

    #pragma omp parallel
    {
        Evaluator eval(context);
        #pragma omp for schedule(static)
        for (size_t d = 1; d < baby; ++d) {
            eval.rotate_rows(query, static_cast<int>(d), gk, qs[d]);
        }
    }
    return qs;
}

/* -------------------  主流程  ------------------- */
void vector_match(
    size_t POLY_DEGREE,
    size_t NUM_EMBEDDINGS,
    size_t DIM,
    size_t OMP_THREADS,
    size_t PlainModulus,
    EmbMode data_mode,
    EmbMode query_mode,
    uint32_t data_seed,
    uint32_t query_seed,
    long long SCALE_INT,    // 可调固定小数尺度
    size_t TOP_K,
    bool L2_NORMALIZE,
    Metric metric = Metric::Cosine
){
    /* 0. 先生成明文数据与查询，用于估算模数 */
    omp_set_num_threads(static_cast<int>(OMP_THREADS));

    auto embeddings = generate_embeddings(NUM_EMBEDDINGS, DIM, data_mode, data_seed, /*scale*/1.0, /*sparsity*/0.9, L2_NORMALIZE);
    vector<double> qvecF;
    size_t random_idx = 0;

    if (data_mode == EmbMode::Test) {
        // 测试：查询为全 1（长度 DIM）
        qvecF.assign(DIM, 1.0);
    } else {
        // 正常：从库里随机挑一条作查询
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(0, NUM_EMBEDDINGS - 1);
        random_idx = dis(gen);
        qvecF = embeddings[random_idx];
    }

    // 计算 Top-K（真实相似度排名）（Test 模式无需使用）
    set<size_t> true_top_k;
    if (data_mode != EmbMode::Test) {
        if (metric == Metric::Euclid) {
            true_top_k = topk_by_euclid(qvecF, embeddings, TOP_K);
        } else if (L2_NORMALIZE) {
            true_top_k = get_top_k_indices(qvecF, embeddings, TOP_K);
        } else {
            vector<double> b_norms = compute_l2_norms(embeddings);
            vector<pair<double,size_t>> sims; sims.reserve(NUM_EMBEDDINGS);
            for (size_t i = 0; i < NUM_EMBEDDINGS; ++i) {
                long double dot = 0.0L;
                for (size_t d = 0; d < DIM; ++d) dot += (long double)qvecF[d]*(long double)embeddings[i][d];
                long double score = dot / (long double)b_norms[i]; // 只除 ‖b‖
                sims.emplace_back((double)score, i);
            }
            sort(sims.begin(), sims.end(), [](auto &a, auto &b){ return a.first > b.first; });
            for (size_t k = 0; k < TOP_K && k < sims.size(); ++k) true_top_k.insert(sims[k].second);
        }
    }

    // 计算每端放大后的最大幅度（用于边界估计）
    // DB 端缩放后单元素上界
    auto absmax_scaled = [&](const vector<vector<double>>& M)->long long{
        long double m = 0.0L;
        for (auto &row : M)
            for (double x : row)
                m = max(m, fabsl((long double)x * (long double)SCALE_INT));
        long double lim = min<long double>(m, (long double)LLONG_MAX);
        return (long long)ceil(lim);
    };

    long long A = max(1LL, absmax_scaled(embeddings));
    long long B = A;

    // 计算 K：若 L2_NORMALIZE=false，需要 K = ceil(inv_scale_int * max_i 1/||b_i||)，否则 K=1
    long long K = 1;
    int t_bits = 0;
    vector<double> b_norms_for_inv;
    if (metric == Metric::Cosine) {
        if (!L2_NORMALIZE) {
            b_norms_for_inv = compute_l2_norms(embeddings);
            long double inv_max = 0.0L;
            for (double nb : b_norms_for_inv) inv_max = max(inv_max, 1.0L / (long double)nb);
            long long inv_scale_int = SCALE_INT; // 复用你的 SCALE_INT 作为倒数量化，或改成你想要的因子
            long long K_tmp = (long long)ceil((long double)inv_scale_int * inv_max);
            if (K_tmp < 1) K_tmp = 1;
            K = K_tmp;
        }
        t_bits = choose_t_bits_from_bounds_min_extra(A, B, DIM, K, /*extra_bits*/3);
    } else {
        t_bits = choose_t_bits_for_euclid(A, B, DIM, /*extra_bits*/3);
    }
    
    cerr << "[info] estimated t_bits = " << t_bits
        << " (A=" << A << ", D=" << DIM << ", K=" << K << ")\n";

    if (data_mode != EmbMode::Test) {
        PlainModulus = max<size_t>(PlainModulus, (size_t)t_bits);
    }

    /* 1. SEAL 上下文（这里固定 t_bits=30，足以覆盖 Test）*/
    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(POLY_DEGREE);
    parms.set_coeff_modulus(CoeffModulus::BFVDefault(POLY_DEGREE));
    parms.set_plain_modulus(PlainModulus::Batching(POLY_DEGREE, PlainModulus));
    SEALContext context(parms);

    KeyGenerator keygen(context);
    PublicKey  pk;  keygen.create_public_key(pk);
    SecretKey  sk  = keygen.secret_key();
    GaloisKeys gk;  keygen.create_galois_keys(gk);

    Encryptor  encryptor(context, pk);
    Decryptor  decryptor(context, sk);
    BatchEncoder encoder(context);

    const size_t degreeN = parms.poly_modulus_degree();
    assert(degreeN == POLY_DEGREE && "degree mismatch");

    const uint64_t plain_modulus_t = context.first_context_data()->parms().plain_modulus().value();

    Plaintext qnorm2_pt;
    vector<Plaintext> bnorm2_row_pts;
    if (metric == Metric::Euclid) {
        // qnorm2_pt
        __int128 acc_q = 0;
        for (size_t d = 0; d < DIM; ++d) {
            long long qs = llround((long double)qvecF[d] * (long double)SCALE_INT);
            acc_q += (__int128)qs * (__int128)qs;
        }
        long long q2 = (long long)min<__int128>(acc_q, LLONG_MAX);
        vector<uint64_t> slots(degreeN, map_to_Zt(q2, plain_modulus_t));
        BatchEncoder enc(context);
        enc.encode(slots, qnorm2_pt);

        // bnorm2_row_pts[i]
        bnorm2_row_pts = pack_bnorm2_per_row(context, embeddings,
                                            degreeN, DIM, plain_modulus_t, SCALE_INT);
    }

    vector<Plaintext> invnorm_row_pts; // Cosine非归一化需要
    if (metric == Metric::Cosine && !L2_NORMALIZE) {
        if (b_norms_for_inv.empty()) b_norms_for_inv = compute_l2_norms(embeddings);
        long long inv_scale_int = SCALE_INT;
        invnorm_row_pts = pack_inv_norms_per_row(context, b_norms_for_inv, degreeN, plain_modulus_t, inv_scale_int);
    }

    /* 2. encode 数据库（对角线+BSGS 打包）*/
    const auto [baby, giant] = plan_bsgs(DIM);
    const size_t fd = ceil_div(NUM_EMBEDDINGS, degreeN); // 数据“行”数

    int db_coeff = (metric == Metric::Euclid ? -2 : +1);
    auto db = encode_database_diagonal_parallel(context, embeddings, degreeN, DIM,
                                            SCALE_INT, plain_modulus_t, db_coeff);


    /* 3. plaintext → NTT 预处理（并行） */
    for (size_t i = 0; i < db.size(); ++i) {
        #pragma omp parallel
        {
            Evaluator eval(context);
            #pragma omp for schedule(static)
            for (size_t j = 0; j < db[i].size(); ++j)
                eval.transform_to_ntt(db[i][j], context.first_parms_id(), db[i][j]);
        }
    }

    /* 4. 加密查询 + baby-step 旋转 & NTT */
    Ciphertext query_ct = encrypt_query_from_vec(encoder, encryptor, degreeN, DIM, qvecF, SCALE_INT, plain_modulus_t);

    auto t0 = chrono::high_resolution_clock::now();

    auto queryset = make_queryset(context, query_ct, gk, baby);

    #pragma omp parallel
    {
        Evaluator eval(context);
        #pragma omp for schedule(static)
        for (size_t i = 0; i < queryset.size(); ++i)
            eval.transform_to_ntt_inplace(queryset[i]); // 密文 → NTT
    }

    /* 5. BSGS 乘加*/
    vector<Ciphertext> result(fd);

    for (size_t i = 0; i < fd; ++i) {
        vector<Ciphertext> row_part(giant);

        #pragma omp parallel for schedule(static)
        for (size_t j = 0; j < giant; ++j) {
            try {
                Evaluator eval(context);
                vector<Ciphertext> local_tmp; local_tmp.reserve(baby);
                for (size_t k = 0; k < baby; ++k) {
                    size_t d = j * baby + k;
                    if (d >= DIM) break;

                    Ciphertext prod;
                    eval.multiply_plain(queryset[k], db[i][d], prod);
                    local_tmp.push_back(std::move(prod));
                }

                if (local_tmp.empty()) continue;
                Ciphertext acc = local_tmp[0];
                for (size_t t = 1; t < local_tmp.size(); ++t) eval.add_inplace(acc, local_tmp[t]);

                eval.transform_from_ntt_inplace(acc);
                if (j) eval.rotate_rows_inplace(acc, static_cast<int>(j * baby), gk);
                row_part[j] = std::move(acc);

            } catch (const std::exception &e) {
                #pragma omp critical
                std::cerr << "[thread " << omp_get_thread_num() << "] " << e.what() << std::endl;
                std::abort();
            }
        }

        Evaluator eval(context);
        Ciphertext row_acc;
        bool init = false;
        for (size_t j = 0; j < giant; ++j) {
            if (row_part[j].parms_id() == parms_id_zero) continue;
            if (!init) { row_acc = row_part[j]; init = true; }
            else eval.add_inplace(row_acc, row_part[j]);
        }

        if (metric == Metric::Cosine) {
            if (!L2_NORMALIZE && init) {
                // row_acc（普通域） ×（普通域倒数明文）
                eval.multiply_plain_inplace(row_acc, invnorm_row_pts[i]);
            }
        } else {
            // row_acc 现在已经是 -2⟨q,b⟩（因为 db_coeff=-2）
            eval.add_plain_inplace(row_acc, qnorm2_pt);        // + ||q||^2 (scaled)
            eval.add_plain_inplace(row_acc, bnorm2_row_pts[i]); // + ||b||^2 (scaled)
        }
        
        result[i] = std::move(row_acc);
    }

    auto t1 = chrono::high_resolution_clock::now();
    auto ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
    cout << "Compute time: " << ms << " ms\n";

    /* 6. 解密结果 + 还原 */
    if (data_mode == EmbMode::Test) {
        Plaintext res_pt; decryptor.decrypt(result[0], res_pt);
        vector<uint64_t> u; encoder.decode(res_pt, u);
        const long double scale_sq = (long double)SCALE_INT * (long double)SCALE_INT;

        cout << "[TEST] first 16: ";
        for (size_t i = 0; i < 16; ++i) {
            long long s = center_lift_from_Zt(u[i], plain_modulus_t);
            long double real_v = (long double)s / scale_sq;
            cout << (double)real_v << (i+1<16? ", ":"\n");
        }
    } else {
        /* 常规 Top-K 验证 */
        vector<pair<double, size_t>> all_scores;
        all_scores.reserve(NUM_EMBEDDINGS);

        const long double scale_sq = (long double)SCALE_INT * (long double)SCALE_INT;
        const long double post_div =
        (metric == Metric::Cosine)
        ? ( L2_NORMALIZE ? ( (long double)SCALE_INT * SCALE_INT )
                        : ( (long double)SCALE_INT * SCALE_INT * (long double)SCALE_INT ) )
        : ( (long double)SCALE_INT * SCALE_INT );   // Euclid
        size_t global_base = 0;

        for (size_t i = 0; i < fd; ++i) {
            if (i >= result.size()) break;
            if (result[i].parms_id() == parms_id_zero) { global_base += degreeN; continue; }

            Plaintext pt; decryptor.decrypt(result[i], pt);
            vector<uint64_t> slots_u64; encoder.decode(pt, slots_u64);

            for (size_t j = 0; j < slots_u64.size(); ++j) {
                size_t gid = global_base + j;
                if (gid >= NUM_EMBEDDINGS) break;

                long long signed_v = center_lift_from_Zt(slots_u64[j], plain_modulus_t);
                long double real_v = (long double)signed_v / post_div; // 归一化：得到 cos；非归一化：得到 dot/||b||
                if (metric == Metric::Euclid) real_v = -real_v;
                all_scores.emplace_back((double)real_v, gid);
            }
            global_base += degreeN;
        }

        sort(all_scores.begin(), all_scores.end(),
            [](const auto &a, const auto &b){ return a.first > b.first; });

        set<size_t> dec_top_k;
        for (size_t i = 0; i < TOP_K && i < all_scores.size(); ++i)
            dec_top_k.insert(all_scores[i].second);

        size_t intersect_cnt = 0;
        for (auto id : dec_top_k) if (true_top_k.count(id)) ++intersect_cnt;

        cout << "Top-" << TOP_K << " overlap (HE vs Plain): "
            << intersect_cnt << " / " << TOP_K << "\n";
    }
}

int main2() {
    
    vector_match(4096, 4096, 1024, 64, 20,
                EmbMode::Uniform, EmbMode::Uniform,
                2025, 7, 1024, 100, true, Metric::Cosine);  // 256可能就快到极限了
    vector_match(4096, 4096, 1024, 64, 20,
                EmbMode::Uniform, EmbMode::Uniform,
                2025, 7, 512, 100, true, Metric::Cosine);  // 256可能就快到极限了
    vector_match(4096, 4096, 1024, 64, 20,
                EmbMode::Uniform, EmbMode::Uniform,
                2025, 7, 256, 100, true, Metric::Cosine);  // 256可能就快到极限了
    vector_match(4096, 4096, 1024, 64, 20,
                EmbMode::Uniform, EmbMode::Uniform,
                2025, 7, 128, 100, true, Metric::Cosine);  // 256可能就快到极限了
    vector_match(4096, 4096, 1024, 64, 20,
                EmbMode::Uniform, EmbMode::Uniform,
                2025, 7, 64, 100, true, Metric::Cosine);  // 256可能就快到极限了
// vector_match(8192, 8192, 1024, 64, 25,
    //                       EmbMode::Uniform, EmbMode::Uniform,
    //                       2025, 7, 1024, 100);  
    return 0;
}

// TODO
// 1. 增加bool normalize参数，控制是否归一化 (已经完成，几乎没有可用性啊，除法的大小没法控制，导致放大太高了)
// 2. 设计基于欧式距离的实现 (已经完成)
// 3. PK-EVK的实现（hard）
// 4. 将设计移植到CKKS方案下（不着急，BFV够用了）