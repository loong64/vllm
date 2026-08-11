#ifndef CPU_ATTN_LOONGARCH_DEFS_HPP
#define CPU_ATTN_LOONGARCH_DEFS_HPP

#if !defined(__loongarch_sx) && !defined(__loongarch_asx)
  #error "cpu_attn_loongarch_defs.hpp requires -mlsx or -mlasx"
#endif

#include "cpu_attn_impl.hpp"

namespace cpu_attention {

namespace {

constexpr int32_t kBlockSizeAlignment = 32;
constexpr int32_t kHeadSizeAlignment = 32;
constexpr int32_t kMaxQHeadNumPerIteration = 16;

template <typename kv_cache_t>
FORCE_INLINE vec_op::FP32Vec8 load_row8_as_f32(const kv_cache_t* ptr) {
  if constexpr (std::is_same_v<kv_cache_t, float>) {
    return vec_op::FP32Vec8(ptr);
  } else if constexpr (std::is_same_v<kv_cache_t, c10::Half>) {
    return vec_op::FP32Vec8(vec_op::FP16Vec8(ptr));
  } else {
    return vec_op::FP32Vec8(vec_op::BF16Vec8(ptr));
  }
}

template <int32_t M, typename kv_cache_t>
FORCE_INLINE void gemm_micro_loongarch_Mx8(
    const float* __restrict a, const kv_cache_t* __restrict b,
    float* __restrict c, int64_t lda, int64_t ldb, int64_t ldc, int32_t k_size,
    bool accumulate) {
  static_assert(1 <= M && M <= 8, "M must be in [1, 8]");

  vec_op::FP32Vec8 acc[M];
  for (int32_t m = 0; m < M; ++m) {
    acc[m] = accumulate ? vec_op::FP32Vec8(c + m * ldc) : vec_op::FP32Vec8();
  }
  for (int32_t k = 0; k < k_size; ++k) {
    const auto b_vec = load_row8_as_f32(b + k * ldb);
    for (int32_t m = 0; m < M; ++m) {
      acc[m] = acc[m] + vec_op::FP32Vec8(a[m * lda + k]) * b_vec;
    }
  }
  for (int32_t m = 0; m < M; ++m) {
    acc[m].save(c + m * ldc);
  }
}

template <int32_t N, typename kv_cache_t>
FORCE_INLINE void gemm_macro_loongarch_Mx8(
    const float* __restrict a, const kv_cache_t* __restrict b,
    float* __restrict c, int32_t m_size, int32_t k_size, int64_t lda,
    int64_t ldb, int64_t ldc, bool accumulate) {
  static_assert(N % 8 == 0, "N must be a multiple of 8");

  for (int32_t m = 0; m < m_size;) {
    const int32_t mb = (m_size - m >= 8)   ? 8
                       : (m_size - m >= 4) ? 4
                       : (m_size - m >= 2) ? 2
                                           : 1;
    for (int32_t n = 0; n < N; n += 8) {
      const auto* b_tile = b + n;
      float* c_tile = c + m * ldc + n;
      switch (mb) {
        case 8:
          gemm_micro_loongarch_Mx8<8>(a + m * lda, b_tile, c_tile, lda, ldb,
                                       ldc, k_size, accumulate);
          break;
        case 4:
          gemm_micro_loongarch_Mx8<4>(a + m * lda, b_tile, c_tile, lda, ldb,
                                       ldc, k_size, accumulate);
          break;
        case 2:
          gemm_micro_loongarch_Mx8<2>(a + m * lda, b_tile, c_tile, lda, ldb,
                                       ldc, k_size, accumulate);
          break;
        default:
          gemm_micro_loongarch_Mx8<1>(a + m * lda, b_tile, c_tile, lda, ldb,
                                       ldc, k_size, accumulate);
          break;
      }
    }
    m += mb;
  }
}

template <typename kv_cache_t>
class TileGemmLoongArch {
 public:
  template <AttentionGemmPhase phase, int32_t k_size>
  FORCE_INLINE static void gemm(const int32_t m_size,
                                float* __restrict__ a_tile,
                                kv_cache_t* __restrict__ b_tile,
                                float* __restrict__ c_tile,
                                const int64_t lda, const int64_t ldb,
                                const int64_t ldc, const int32_t,
                                const int32_t dynamic_k_size,
                                const bool accum_c) {
    constexpr int32_t n = phase == AttentionGemmPhase::QK
                              ? kBlockSizeAlignment
                              : kHeadSizeAlignment;
    gemm_macro_loongarch_Mx8<n>(
        a_tile, b_tile, c_tile, m_size,
        phase == AttentionGemmPhase::QK ? k_size : dynamic_k_size, lda, ldb,
        ldc, accum_c);
  }
};

}  // namespace

}  // namespace cpu_attention

#endif  // CPU_ATTN_LOONGARCH_DEFS_HPP
