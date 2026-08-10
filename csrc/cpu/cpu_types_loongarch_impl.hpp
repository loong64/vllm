#ifndef CPU_TYPES_LOONGARCH_IMPL_HPP
#define CPU_TYPES_LOONGARCH_IMPL_HPP

#include "cpu_types_loongarch_defs.hpp"

namespace vec_op {

struct FP16Vec8 : Vec<FP16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;
  f16x8_t reg;

  explicit FP16Vec8(const void* ptr)
      : reg(*static_cast<const f16x8_t*>(ptr)) {}
  explicit FP16Vec8(const FP32Vec8&);

  void save(void* ptr) const { *static_cast<f16x8_t*>(ptr) = reg; }
};

struct FP16Vec16 : Vec<FP16Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;
  f16x16_t reg;

  explicit FP16Vec16(const void* ptr)
      : reg(*static_cast<const f16x16_t*>(ptr)) {}
  explicit FP16Vec16(const FP32Vec16&);

  void save(void* ptr) const { *static_cast<f16x16_t*>(ptr) = reg; }
  void save(void* ptr, int count) const {
    std::memcpy(ptr, reg.val,
                std::min(count, VEC_ELEM_NUM) * sizeof(uint16_t));
  }
};

struct BF16Vec8 : Vec<BF16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;
  f16x8_t reg;

  explicit BF16Vec8(const void* ptr)
      : reg(*static_cast<const f16x8_t*>(ptr)) {}
  explicit BF16Vec8(const FP32Vec8&);

  void save(void* ptr) const { *static_cast<f16x8_t*>(ptr) = reg; }
};

struct BF16Vec16 : Vec<BF16Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;
  f16x16_t reg;

  explicit BF16Vec16(const void* ptr)
      : reg(*static_cast<const f16x16_t*>(ptr)) {}
  explicit BF16Vec16(const FP32Vec16&);

  void save(void* ptr) const { *static_cast<f16x16_t*>(ptr) = reg; }
  void save(void* ptr, int count) const {
    std::memcpy(ptr, reg.val,
                std::min(count, VEC_ELEM_NUM) * sizeof(uint16_t));
  }
};

struct BF16Vec32 : Vec<BF16Vec32> {
  constexpr static int VEC_ELEM_NUM = 32;
  f16x32_t reg;

  explicit BF16Vec32(const void* ptr)
      : reg(*static_cast<const f16x32_t*>(ptr)) {}
  explicit BF16Vec32(f16x32_t data) : reg(data) {}
  explicit BF16Vec32(BF16Vec8& data) {
    unroll_loop<int, VEC_ELEM_NUM>(
        [&](int i) { reg.val[i] = data.reg.val[i % 8]; });
  }
  explicit BF16Vec32(const uint8_t*, fp8_e4m3_tag) : reg{} {}
  explicit BF16Vec32(const uint8_t*, fp8_e5m2_tag) : reg{} {}

  void save(void* ptr) const { *static_cast<f16x32_t*>(ptr) = reg; }
};

struct FP32Vec4 : Vec<FP32Vec4> {
  constexpr static int VEC_ELEM_NUM = 4;
  f32x4_t reg;

  explicit FP32Vec4(float v) {
    for (auto& x : reg.val) {
      x = v;
    }
  }
  explicit FP32Vec4() : reg{} {}
  explicit FP32Vec4(const float* ptr)
      : reg(*reinterpret_cast<const f32x4_t*>(ptr)) {}
  explicit FP32Vec4(f32x4_t data) : reg(data) {}
  FP32Vec4(const FP32Vec4& data) : reg(data.reg) {}
};

struct FP32Vec8 : Vec<FP32Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;
  f32x8_t reg;

  explicit FP32Vec8(float v) {
    for (auto& x : reg.val) {
      x = v;
    }
  }
  explicit FP32Vec8() : reg{} {}
  explicit FP32Vec8(const float* ptr)
      : reg(*reinterpret_cast<const f32x8_t*>(ptr)) {}
  explicit FP32Vec8(f32x8_t data) : reg(data) {}
  FP32Vec8(const FP32Vec8& data) : reg(data.reg) {}
  explicit FP32Vec8(const FP16Vec8& v) {
    for (int i = 0; i < VEC_ELEM_NUM; ++i) {
      reg.val[i] = fp16_to_float(v.reg.val[i]);
    }
  }
  FP32Vec8(const BF16Vec8& v) {
    for (int i = 0; i < VEC_ELEM_NUM; ++i) {
      reg.val[i] = bf16_to_float(v.reg.val[i]);
    }
  }

  float reduce_sum() const {
    float result = 0.0f;
    for (float value : reg.val) {
      result += value;
    }
    return result;
  }
  FP32Vec8 exp() const {
    f32x8_t result;
    for (int i = 0; i < VEC_ELEM_NUM; ++i) {
      result.val[i] = expf(reg.val[i]);
    }
    return FP32Vec8(result);
  }
  FP32Vec8 tanh() const {
    f32x8_t result;
    for (int i = 0; i < VEC_ELEM_NUM; ++i) {
      result.val[i] = tanhf(reg.val[i]);
    }
    return FP32Vec8(result);
  }
  FP32Vec8 er() const {
    f32x8_t result;
    for (int i = 0; i < VEC_ELEM_NUM; ++i) {
      result.val[i] = erf(reg.val[i]);
    }
    return FP32Vec8(result);
  }
  FP32Vec8 operator*(const FP32Vec8& b) const {
    f32x8_t result;
    binary<BinaryOp::Mul, VEC_ELEM_NUM>(result.val, reg.val, b.reg.val);
    return FP32Vec8(result);
  }
  FP32Vec8 operator+(const FP32Vec8& b) const {
    f32x8_t result;
    binary<BinaryOp::Add, VEC_ELEM_NUM>(result.val, reg.val, b.reg.val);
    return FP32Vec8(result);
  }
  FP32Vec8 operator-(const FP32Vec8& b) const {
    f32x8_t result;
    binary<BinaryOp::Sub, VEC_ELEM_NUM>(result.val, reg.val, b.reg.val);
    return FP32Vec8(result);
  }
  FP32Vec8 operator/(const FP32Vec8& b) const {
    f32x8_t result;
    binary<BinaryOp::Div, VEC_ELEM_NUM>(result.val, reg.val, b.reg.val);
    return FP32Vec8(result);
  }

  void save(void* ptr) const { *static_cast<f32x8_t*>(ptr) = reg; }
};

struct FP32Vec16 : Vec<FP32Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;
  f32x16_t reg;

  explicit FP32Vec16(float v) {
    for (auto& x : reg.val) {
      x = v;
    }
  }
  explicit FP32Vec16() : reg{} {}
  explicit FP32Vec16(const float* ptr)
      : reg(*reinterpret_cast<const f32x16_t*>(ptr)) {}
  explicit FP32Vec16(f32x16_t data) : reg(data) {}
  FP32Vec16(const FP32Vec4& data) {
    for (int i = 0; i < VEC_ELEM_NUM; ++i) {
      reg.val[i] = data.reg.val[i % FP32Vec4::VEC_ELEM_NUM];
    }
  }
  FP32Vec16(const FP32Vec8& data) {
    for (int i = 0; i < VEC_ELEM_NUM; ++i) {
      reg.val[i] = data.reg.val[i % FP32Vec8::VEC_ELEM_NUM];
    }
  }
  FP32Vec16(const FP32Vec16& data) : reg(data.reg) {}
  explicit FP32Vec16(const FP16Vec16& v) {
    for (int i = 0; i < VEC_ELEM_NUM; ++i) {
      reg.val[i] = fp16_to_float(v.reg.val[i]);
    }
  }
  explicit FP32Vec16(const BF16Vec16& v) {
    for (int i = 0; i < VEC_ELEM_NUM; ++i) {
      reg.val[i] = bf16_to_float(v.reg.val[i]);
    }
  }
  explicit FP32Vec16(const FP16Vec8& v) : FP32Vec16(FP32Vec8(v)) {}
  FP32Vec16(const BF16Vec8& v) : FP32Vec16(FP32Vec8(v)) {}
  explicit FP32Vec16(const BF16Vec32&, int) : reg{} {}

  FP32Vec16 operator*(const FP32Vec16& b) const {
    f32x16_t result;
    binary<BinaryOp::Mul, VEC_ELEM_NUM>(result.val, reg.val, b.reg.val);
    return FP32Vec16(result);
  }
  FP32Vec16 operator+(const FP32Vec16& b) const {
    f32x16_t result;
    binary<BinaryOp::Add, VEC_ELEM_NUM>(result.val, reg.val, b.reg.val);
    return FP32Vec16(result);
  }
  FP32Vec16 operator-(const FP32Vec16& b) const {
    f32x16_t result;
    binary<BinaryOp::Sub, VEC_ELEM_NUM>(result.val, reg.val, b.reg.val);
    return FP32Vec16(result);
  }
  FP32Vec16 operator/(const FP32Vec16& b) const {
    f32x16_t result;
    binary<BinaryOp::Div, VEC_ELEM_NUM>(result.val, reg.val, b.reg.val);
    return FP32Vec16(result);
  }
  FP32Vec16 max(const FP32Vec16& b) const {
    f32x16_t result;
    binary<BinaryOp::Max, VEC_ELEM_NUM>(result.val, reg.val, b.reg.val);
    return FP32Vec16(result);
  }
  FP32Vec16 min(const FP32Vec16& b) const {
    f32x16_t result;
    binary<BinaryOp::Min, VEC_ELEM_NUM>(result.val, reg.val, b.reg.val);
    return FP32Vec16(result);
  }
  FP32Vec16 abs() const {
    f32x16_t result;
    for (int i = 0; i < VEC_ELEM_NUM; ++i) {
      result.val[i] = std::abs(reg.val[i]);
    }
    return FP32Vec16(result);
  }
  FP32Vec16 tanh() const {
    f32x16_t result;
    for (int i = 0; i < VEC_ELEM_NUM; ++i) {
      result.val[i] = std::tanh(reg.val[i]);
    }
    return FP32Vec16(result);
  }
  float reduce_sum() const {
    float result = 0.0f;
    for (float value : reg.val) {
      result += value;
    }
    return result;
  }
  float reduce_max() const {
    float result = std::numeric_limits<float>::lowest();
    for (float value : reg.val) {
      result = std::max(result, value);
    }
    return result;
  }
  float reduce_min() const {
    float result = std::numeric_limits<float>::max();
    for (float value : reg.val) {
      result = std::min(result, value);
    }
    return result;
  }
  template <int group_size>
  float reduce_sub_sum(int idx) {
    float result = 0.0f;
    for (int i = 0; i < group_size; ++i) {
      result += reg.val[idx * group_size + i];
    }
    return result;
  }

  void save(void* ptr) const { *static_cast<f32x16_t*>(ptr) = reg; }
};

template <typename T>
struct VecType {
  using vec_type = void;
};
template <typename T>
using vec_t = typename VecType<T>::vec_type;
template <>
struct VecType<float> {
  using vec_type = FP32Vec8;
};
template <>
struct VecType<c10::Half> {
  using vec_type = FP16Vec8;
};
template <>
struct VecType<c10::BFloat16> {
  using vec_type = BF16Vec8;
};

template <typename T>
void storeFP32(float v, T* ptr) {
  *ptr = v;
}
template <>
inline void storeFP32<c10::Half>(float v, c10::Half* ptr) {
  *reinterpret_cast<uint16_t*>(ptr) = float_to_fp16(v);
}
template <>
inline void storeFP32<c10::BFloat16>(float v, c10::BFloat16* ptr) {
  *ptr = static_cast<c10::BFloat16>(v);
}

inline FP16Vec8::FP16Vec8(const FP32Vec8& v) {
  for (int i = 0; i < VEC_ELEM_NUM; ++i) {
    reg.val[i] = float_to_fp16(v.reg.val[i]);
  }
}
inline FP16Vec16::FP16Vec16(const FP32Vec16& v) {
  for (int i = 0; i < VEC_ELEM_NUM; ++i) {
    reg.val[i] = float_to_fp16(v.reg.val[i]);
  }
}
inline BF16Vec8::BF16Vec8(const FP32Vec8& v) {
  for (int i = 0; i < VEC_ELEM_NUM; ++i) {
    reg.val[i] = float_to_bf16(v.reg.val[i]);
  }
}
inline BF16Vec16::BF16Vec16(const FP32Vec16& v) {
  for (int i = 0; i < VEC_ELEM_NUM; ++i) {
    reg.val[i] = float_to_bf16(v.reg.val[i]);
  }
}

inline void fma(FP32Vec16& acc, FP32Vec16& a, FP32Vec16& b) {
  acc = acc + a * b;
}
inline void prefetch(const void* addr) { __builtin_prefetch(addr, 0, 3); }

}  // namespace vec_op

#endif  // CPU_TYPES_LOONGARCH_IMPL_HPP
