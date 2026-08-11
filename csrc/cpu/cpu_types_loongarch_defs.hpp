#ifndef CPU_TYPES_LOONGARCH_DEFS_HPP
#define CPU_TYPES_LOONGARCH_DEFS_HPP

#if !defined(__loongarch_sx) && !defined(__loongarch_asx)
  #error "cpu_types_loongarch_defs.hpp requires -mlsx or -mlasx"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

#include <lsxintrin.h>
#if defined(__loongarch_asx)
  #include <lasxintrin.h>
#endif
#include <torch/all.h>

#include "float_convert.hpp"

namespace vec_op {

struct fp8_e4m3_tag {};
struct fp8_e5m2_tag {};

#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)            \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)    \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#ifndef CPU_OP_GUARD
  #define CPU_KERNEL_GUARD_IN(NAME)
  #define CPU_KERNEL_GUARD_OUT(NAME)
#else
  #define CPU_KERNEL_GUARD_IN(NAME) \
    std::cout << #NAME << " invoked." << std::endl;
  #define CPU_KERNEL_GUARD_OUT(NAME) std::cout << #NAME << " exit." << std::endl;
#endif

#define FORCE_INLINE __attribute__((always_inline)) inline

struct f16x8_t {
  uint16_t val[8];
};
struct f16x16_t {
  uint16_t val[16];
};
struct f16x32_t {
  uint16_t val[32];
};
struct f32x4_t {
  float val[4];
};
struct f32x8_t {
  float val[8];
};
struct f32x16_t {
  float val[16];
};

namespace {

template <typename T, T... indexes, typename F>
constexpr void unroll_loop_item(std::integer_sequence<T, indexes...>, F&& f) {
  (f(std::integral_constant<T, indexes>{}), ...);
}

enum class BinaryOp { Add, Sub, Mul, Div, Min, Max };

template <BinaryOp op>
FORCE_INLINE __m128 lsx_binary(__m128 a, __m128 b) {
  if constexpr (op == BinaryOp::Add) {
    return __lsx_vfadd_s(a, b);
  } else if constexpr (op == BinaryOp::Sub) {
    return __lsx_vfsub_s(a, b);
  } else if constexpr (op == BinaryOp::Mul) {
    return __lsx_vfmul_s(a, b);
  } else if constexpr (op == BinaryOp::Div) {
    return __lsx_vfdiv_s(a, b);
  } else if constexpr (op == BinaryOp::Min) {
    return __lsx_vfmin_s(a, b);
  } else {
    return __lsx_vfmax_s(a, b);
  }
}

#if defined(__loongarch_asx)
template <BinaryOp op>
FORCE_INLINE __m256 lasx_binary(__m256 a, __m256 b) {
  if constexpr (op == BinaryOp::Add) {
    return __lasx_xvfadd_s(a, b);
  } else if constexpr (op == BinaryOp::Sub) {
    return __lasx_xvfsub_s(a, b);
  } else if constexpr (op == BinaryOp::Mul) {
    return __lasx_xvfmul_s(a, b);
  } else if constexpr (op == BinaryOp::Div) {
    return __lasx_xvfdiv_s(a, b);
  } else if constexpr (op == BinaryOp::Min) {
    return __lasx_xvfmin_s(a, b);
  } else {
    return __lasx_xvfmax_s(a, b);
  }
}
#endif

template <BinaryOp op, int count>
FORCE_INLINE void binary(float* out, const float* lhs, const float* rhs) {
  int i = 0;
#if defined(__loongarch_asx)
  for (; i + 8 <= count; i += 8) {
    const auto a = (__m256)__lasx_xvld(const_cast<float*>(lhs + i), 0);
    const auto b = (__m256)__lasx_xvld(const_cast<float*>(rhs + i), 0);
    __lasx_xvst((__m256i)lasx_binary<op>(a, b), out + i, 0);
  }
#endif
  for (; i + 4 <= count; i += 4) {
    const auto a = (__m128)__lsx_vld(const_cast<float*>(lhs + i), 0);
    const auto b = (__m128)__lsx_vld(const_cast<float*>(rhs + i), 0);
    __lsx_vst((__m128i)lsx_binary<op>(a, b), out + i, 0);
  }
  for (; i < count; ++i) {
    if constexpr (op == BinaryOp::Add) {
      out[i] = lhs[i] + rhs[i];
    } else if constexpr (op == BinaryOp::Sub) {
      out[i] = lhs[i] - rhs[i];
    } else if constexpr (op == BinaryOp::Mul) {
      out[i] = lhs[i] * rhs[i];
    } else if constexpr (op == BinaryOp::Div) {
      out[i] = lhs[i] / rhs[i];
    } else if constexpr (op == BinaryOp::Min) {
      out[i] = std::min(lhs[i], rhs[i]);
    } else {
      out[i] = std::max(lhs[i], rhs[i]);
    }
  }
}

}  // namespace

template <typename T, T count, typename F,
          typename = std::enable_if_t<std::is_invocable_v<F, T>>>
constexpr void unroll_loop(F&& f) {
  unroll_loop_item(std::make_integer_sequence<T, count>{}, std::forward<F>(f));
}

template <typename T>
struct Vec {
  constexpr static int get_elem_num() { return T::VEC_ELEM_NUM; }
};

struct FP32Vec8;
struct FP32Vec16;

}  // namespace vec_op

#endif  // CPU_TYPES_LOONGARCH_DEFS_HPP
