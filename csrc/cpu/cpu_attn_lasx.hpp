#ifndef CPU_ATTN_LASX_HPP
#define CPU_ATTN_LASX_HPP

#if !defined(__loongarch_asx)
  #error "cpu_attn_lasx.hpp requires -mlasx"
#endif

#include "cpu_attn_loongarch_defs.hpp"

namespace cpu_attention {

template <typename scalar_t, int64_t head_dim, typename kv_cache_scalar_t>
class AttentionImpl<ISA::LASX, scalar_t, head_dim, kv_cache_scalar_t> {
 public:
  using query_t = scalar_t;
  using q_buffer_t = float;
  using kv_cache_t = kv_cache_scalar_t;
  using logits_buffer_t = float;
  using partial_output_buffer_t = float;
  using prob_buffer_t = float;

  constexpr static int64_t BlockSizeAlignment = kBlockSizeAlignment;
  constexpr static int64_t HeadDimAlignment = kHeadSizeAlignment;
  constexpr static int64_t MaxQHeadNumPerIteration = kMaxQHeadNumPerIteration;
  constexpr static int64_t HeadDim = head_dim;
  constexpr static ISA ISAType = ISA::LASX;
  constexpr static bool scale_on_logits = false;

  static_assert(HeadDim % HeadDimAlignment == 0);

  template <template <typename tile_gemm_t> typename attention>
  FORCE_INLINE void execute_attention(DEFINE_CPU_ATTENTION_PARAMS) {
    attention<TileGemmLoongArch<kv_cache_t>> iteration;
    iteration(CPU_ATTENTION_PARAMS);
  }

  constexpr static int64_t k_cache_token_group_stride(const int32_t) {
    return BlockSizeAlignment;
  }
  constexpr static int64_t v_cache_token_group_stride(const int32_t) {
    return head_dim * BlockSizeAlignment;
  }
  constexpr static int64_t v_cache_head_group_stride(const int32_t) {
    return HeadDimAlignment;
  }

  static void copy_q_heads_tile(scalar_t* __restrict__ src,
                                float* __restrict__ q_buffer,
                                const int32_t q_num,
                                const int32_t q_heads_per_kv,
                                const int64_t q_num_stride,
                                const int64_t q_head_stride, float scale) {
    constexpr int32_t count = head_dim / 16;
    using load_vec_t = typename VecTypeTrait<scalar_t>::vec_t;
    const vec_op::FP32Vec16 scale_vec(scale);
    for (int32_t q = 0; q < q_num; ++q) {
      for (int32_t h = 0; h < q_heads_per_kv; ++h) {
        scalar_t* in = src + q * q_num_stride + h * q_head_stride;
        float* out = q_buffer + (q * q_heads_per_kv + h) * head_dim;
        for (int32_t i = 0; i < count; ++i) {
          vec_op::FP32Vec16 value{load_vec_t(in)};
          value = value * scale_vec;
          value.save(out);
          in += 16;
          out += 16;
        }
      }
    }
  }

  static void reshape_and_cache(
      const scalar_t* __restrict__ key, const scalar_t* __restrict__ value,
      kv_cache_t* __restrict__ key_cache, kv_cache_t* __restrict__ value_cache,
      const int64_t* __restrict__ slot_mapping, const int64_t token_num,
      const int64_t key_token_num_stride, const int64_t value_token_num_stride,
      const int64_t head_num, const int64_t key_head_num_stride,
      const int64_t value_head_num_stride, const int64_t,
      const int64_t num_blocks_stride, const int64_t cache_head_num_stride,
      const int64_t block_size, const int64_t, const float = 0.0f,
      const float = 0.0f) {
  #pragma omp parallel for collapse(2)
    for (int64_t token = 0; token < token_num; ++token) {
      for (int64_t head = 0; head < head_num; ++head) {
        const int64_t pos = slot_mapping[token];
        if (pos < 0) {
          continue;
        }
        const int64_t block = pos / block_size;
        const int64_t offset = pos % block_size;
        const scalar_t* key_src =
            key + token * key_token_num_stride + head * key_head_num_stride;
        kv_cache_t* key_dst = key_cache + block * num_blocks_stride +
                              head * cache_head_num_stride + offset;
        for (int64_t i = 0; i < head_dim; ++i) {
          key_dst[i * block_size] = key_src[i];
        }
        const scalar_t* value_src =
            value + token * value_token_num_stride + head * value_head_num_stride;
        kv_cache_t* value_dst = value_cache + block * num_blocks_stride +
                                head * cache_head_num_stride + offset * head_dim;
        std::memcpy(value_dst, value_src, head_dim * sizeof(scalar_t));
      }
    }
  }
};

}  // namespace cpu_attention

#endif  // CPU_ATTN_LASX_HPP
