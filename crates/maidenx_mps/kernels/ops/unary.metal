#include "../metal_utils.metal"
#include <metal_stdlib>
#include <metal_math>

using namespace metal;

template <typename T> T maximum(T x, T y) { return (x > y) ? x : y; }

template <typename T> T minimum(T x, T y) { return (x < y) ? x : y; }

// Improved power function with better handling of edge cases
template<typename T>
T m_pow_int(T base, unsigned int exponent) {
    T result = 1;
    while (exponent > 0) {
        if (exponent & 1) {
            result *= base;
        }
        exponent >>= 1;
        base *= base;
    }
    return result;
}

// Improved power function with better handling of edge cases
float m_pow_float(float base, float exponent) {
    if (exponent == 0.0f) {
        return 1.0f;
    }
    
    if (base == 0.0f) {
        return (exponent > 0.0f) ? 0.0f : INFINITY;
    }
    
    if (base == 1.0f) {
        return 1.0f;
    }
    
    if (exponent == 1.0f) {
        return base;
    }
    
    if (floor(exponent) == exponent) {
        if (exponent >= 0.0f) {
            return m_pow_int(base, (unsigned int)exponent);
        } else {
            return 1.0f / m_pow_int(base, (unsigned int)(-exponent));
        }
    }
    
    if (base < 0.0f) {
        return NAN;
    }
    
    return pow(base, exponent);
}

// Improved exp10 function for better precision
float m_exp10(float x) {
    return exp(x * 2.3025850929940456f); // ln(10) ≈ 2.302585...
}

// Improved tan function for better precision
float m_tan(float x) {
    x = fmod(x, 2 * M_PI_F);
    if (x > M_PI_F) x -= 2 * M_PI_F;
    else if (x < -M_PI_F) x += 2 * M_PI_F;
    
    float halfPi = M_PI_F / 2;
    float eps = 1e-6f;
    
    if (fabs(fabs(x) - halfPi) < eps) {
        return x > 0 ? 1e6f : -1e6f;
    }
    
    return sin(x) / cos(x);
}

#define UNARY_OP_OUTPUT(IN_TYPENAME, OUT_TYPENAME, FN_NAME, FUNC)              \
  kernel void metal_##FN_NAME##_kernel(                                        \
      const device IN_TYPENAME *input [[buffer(0)]],                           \
      device OUT_TYPENAME *output [[buffer(1)]],                               \
      constant size_t& num_els [[buffer(2)]],                                  \
      constant size_t& num_dims [[buffer(3)]],                                 \
      constant size_t *metadata [[buffer(4)]],                                 \
      uint thread_index [[thread_position_in_grid]],                           \
      uint threads_per_grid [[threads_per_grid]])                              \
  {                                                                            \
    for (uint id = thread_index; id < num_els; id += threads_per_grid) {       \
      const constant size_t *dims = metadata;                                  \
      const constant size_t *strides = metadata + num_dims;                    \
      const size_t offset = metadata ? *(metadata + 2 * num_dims) : 0;         \
                                                                               \
      if (metadata == nullptr || is_contiguous(num_dims, dims, strides)) {     \
        IN_TYPENAME x = input ? input[offset + id] : (IN_TYPENAME)output[id];  \
        output[id] = FUNC;                                                     \
      } else {                                                                 \
        unsigned strided_i =                                                   \
            offset + get_strided_index(id, num_dims, dims, strides);           \
        IN_TYPENAME x = input ? input[strided_i] : (IN_TYPENAME)output[id];    \
        output[id] = FUNC;                                                     \
      }                                                                        \
    }                                                                          \
  }

#define UNARY_OP(TYPENAME, FN_NAME, FUNC)                                      \
  UNARY_OP_OUTPUT(TYPENAME, TYPENAME, FN_NAME, FUNC)

#define UNARY_OP_WITH_CONSTANT(IN_TYPENAME, OUT_TYPENAME, FN_NAME, FUNC)       \
  kernel void metal_##FN_NAME##_kernel(                                        \
      const device IN_TYPENAME *input [[buffer(0)]],                           \
      device OUT_TYPENAME *output [[buffer(1)]],                               \
      constant size_t& num_els [[buffer(2)]],                                  \
      constant size_t& num_dims [[buffer(3)]],                                 \
      constant size_t *metadata [[buffer(4)]],                                 \
      constant IN_TYPENAME& const_val [[buffer(5)]],                           \
      uint thread_index [[thread_position_in_grid]],                           \
      uint threads_per_grid [[threads_per_grid]])                              \
  {                                                                            \
    for (uint id = thread_index; id < num_els; id += threads_per_grid) {       \
      const constant size_t *dims = metadata;                                  \
      const constant size_t *strides = metadata + num_dims;                    \
      const size_t offset = metadata ? *(metadata + 2 * num_dims) : 0;         \
                                                                               \
      if (metadata == nullptr || is_contiguous(num_dims, dims, strides)) {     \
        IN_TYPENAME x;                                                         \
        if (input) {                                                           \
          x = input[offset + id];                                              \
        } else {                                                               \
          x = output[id];                                                      \
        }                                                                      \
        output[id] = FUNC;                                                     \
      } else {                                                                 \
        unsigned strided_i =                                                   \
            offset + get_strided_index(id, num_dims, dims, strides);           \
        IN_TYPENAME x;                                                         \
        if (input) {                                                           \
          x = input[strided_i];                                                \
        } else {                                                               \
          x = output[id];                                                      \
        }                                                                      \
        output[id] = FUNC;                                                     \
      }                                                                        \
    }                                                                          \
  }

// Negation operations
UNARY_OP(float, neg_f32, -x);
UNARY_OP(bool, neg_bool, !x);
UNARY_OP(int8_t, neg_i8, -x);
UNARY_OP(int16_t, neg_i16, -x);
UNARY_OP(int32_t, neg_i32, -x);

// Absolute value operations
UNARY_OP(float, abs_f32, abs(x));
UNARY_OP(bool, abs_bool, x);
UNARY_OP(int8_t, abs_i8, abs(x));
UNARY_OP(int16_t, abs_i16, abs(x));
UNARY_OP(int32_t, abs_i32, abs(x));

// Sign operations
UNARY_OP(float, sign_f32, (x > 0) ? 1.0f : ((x < 0) ? -1.0f : 0.0f));
UNARY_OP(bool, sign_bool, x ? 1 : 0);
UNARY_OP(uint8_t, sign_u8, (x > 0) ? 1 : 0);
UNARY_OP(uint16_t, sign_u16, (x > 0) ? 1 : 0);
UNARY_OP(uint32_t, sign_u32, (x > 0) ? 1 : 0);
UNARY_OP(int8_t, sign_i8, (x > 0) ? 1 : ((x < 0) ? -1 : 0));
UNARY_OP(int16_t, sign_i16, (x > 0) ? 1 : ((x < 0) ? -1 : 0));
UNARY_OP(int32_t, sign_i32, (x > 0) ? 1 : ((x < 0) ? -1 : 0));

// Square operations
UNARY_OP(float, square_f32, x * x);
UNARY_OP(bool, square_bool, x);
UNARY_OP(uint8_t, square_u8, x * x);
UNARY_OP(uint16_t, square_u16, x * x);
UNARY_OP(uint32_t, square_u32, x * x);
UNARY_OP(int8_t, square_i8, x * x);
UNARY_OP(int16_t, square_i16, x * x);
UNARY_OP(int32_t, square_i32, x * x);

// Square root operations
UNARY_OP(float, sqrt_f32, sqrt(x));
UNARY_OP(bool, sqrt_bool, x);
UNARY_OP(uint8_t, sqrt_u8, (uint8_t)sqrt(float(x)));
UNARY_OP(uint16_t, sqrt_u16, (uint16_t)sqrt(float(x)));
UNARY_OP(uint32_t, sqrt_u32, (uint32_t)sqrt(float(x)));
UNARY_OP(int8_t, sqrt_i8, (int8_t)sqrt(float(abs(x))));
UNARY_OP(int16_t, sqrt_i16, (int16_t)sqrt(float(abs(x))));
UNARY_OP(int32_t, sqrt_i32, (int32_t)sqrt(float(abs(x))));

// Activation functions
UNARY_OP(float, relu_f32, x > 0 ? x : 0);
UNARY_OP(bool, relu_bool, x);
UNARY_OP(uint8_t, relu_u8, x > 0 ? x : 0);
UNARY_OP(uint16_t, relu_u16, x > 0 ? x : 0);
UNARY_OP(uint32_t, relu_u32, x > 0 ? x : 0);
UNARY_OP(int8_t, relu_i8, x > 0 ? x : 0);
UNARY_OP(int16_t, relu_i16, x > 0 ? x : 0);
UNARY_OP(int32_t, relu_i32, x > 0 ? x : 0);

UNARY_OP(float, sigmoid_f32, 1.0f / (1.0f + exp(-x)));
UNARY_OP(bool, sigmoid_bool, x);

UNARY_OP(float, tanh_f32, tanh(x));
UNARY_OP(bool, tanh_bool, x);

UNARY_OP(float, gelu_f32, 0.5f * x * (1.0f + tanh(0.7978845608028654f * (x + 0.044715f * x * x * x))));

// Trigonometric operations
UNARY_OP(float, sin_f32, sin(x));
UNARY_OP(float, cos_f32, cos(x));
UNARY_OP(float, tan_f32, m_tan(x));

// Logarithmic operations
UNARY_OP(float, ln_f32, log(x));
UNARY_OP(float, log10_f32, log10(x));
UNARY_OP(float, log2_f32, log2(x));

// Exponential operations
UNARY_OP(float, exp_f32, exp(x));
UNARY_OP(float, exp10_f32, m_exp10(x));
UNARY_OP(float, exp2_f32, exp2(x));

// Other operations
UNARY_OP(float, softplus_f32, log(1.0f + exp(x)));
UNARY_OP(float, recip_f32, 1.0f / x);

// Logical operations
UNARY_OP_OUTPUT(float, bool, logical_not_f32, x == 0.0f);
UNARY_OP_OUTPUT(bool, bool, logical_not_bool, !x);
UNARY_OP_OUTPUT(uint8_t, bool, logical_not_u8, x == 0u);
UNARY_OP_OUTPUT(uint16_t, bool, logical_not_u16, x == 0u);
UNARY_OP_OUTPUT(uint32_t, bool, logical_not_u32, x == 0u);
UNARY_OP_OUTPUT(int8_t, bool, logical_not_i8, x == 0);
UNARY_OP_OUTPUT(int16_t, bool, logical_not_i16, x == 0);
UNARY_OP_OUTPUT(int32_t, bool, logical_not_i32, x == 0);

// Half-precision floating point operations
UNARY_OP(half, neg_f16, -x);
UNARY_OP(half, abs_f16, abs(x));
UNARY_OP(half, sign_f16, (x > 0.0h) ? 1.0h : ((x < 0.0h) ? -1.0h : 0.0h));
UNARY_OP(half, square_f16, x * x);
UNARY_OP(half, sqrt_f16, sqrt(x));
UNARY_OP(half, relu_f16, float(x) > 0.0f ? x : 0.0h);
UNARY_OP(half, sigmoid_f16, half(1.0f / (1.0f + exp(float(-x)))));
UNARY_OP(half, tanh_f16, half(tanh(float(x))));
UNARY_OP(half, gelu_f16, half(0.5f * float(x) * (1.0f + tanh(0.7978845608028654f * (float(x) + 0.044715f * float(x) * float(x) * float(x))))));
UNARY_OP(half, sin_f16, half(sin(float(x))));
UNARY_OP(half, cos_f16, half(cos(float(x))));
UNARY_OP(half, tan_f16, half(m_tan(float(x))));
UNARY_OP(half, ln_f16, half(log(float(x))));
UNARY_OP(half, log10_f16, half(log10(float(x))));
UNARY_OP(half, log2_f16, half(log2(float(x))));
UNARY_OP(half, exp_f16, half(exp(float(x))));
UNARY_OP(half, exp10_f16, half(m_exp10(float(x))));
UNARY_OP(half, exp2_f16, half(exp2(float(x))));
UNARY_OP(half, softplus_f16, half(log(1.0f + exp(float(x)))));
UNARY_OP(half, recip_f16, half(1.0f / float(x)));

UNARY_OP_OUTPUT(half, bool, logical_not_f16, float(x) == 0.0f);

// BFloat16 operations
UNARY_OP(bfloat, neg_bf16, -x);
UNARY_OP(bfloat, abs_bf16, bfloat(abs(float(x))));
UNARY_OP(bfloat, sign_bf16, (x > 0.0bf) ? 1.0bf : ((x < 0.0bf) ? -1.0bf : 0.0bf));
UNARY_OP(bfloat, square_bf16, x * x);
UNARY_OP(bfloat, sqrt_bf16, bfloat(sqrt(float(x))));
UNARY_OP(bfloat, relu_bf16, x > 0.0bf ? x : 0.0bf);
UNARY_OP(bfloat, sigmoid_bf16, bfloat(1.0f / (1.0f + exp(-float(x)))));
UNARY_OP(bfloat, tanh_bf16, bfloat(tanh(float(x))));
UNARY_OP(bfloat, gelu_bf16, bfloat(0.5f * float(x) * (1.0f + tanh(0.7978845608028654f * (float(x) + 0.044715f * float(x) * float(x) * float(x))))));
UNARY_OP(bfloat, sin_bf16, bfloat(sin(float(x))));
UNARY_OP(bfloat, cos_bf16, bfloat(cos(float(x))));
UNARY_OP(bfloat, tan_bf16, bfloat(m_tan(float(x))));
UNARY_OP(bfloat, ln_bf16, bfloat(log(float(x))));
UNARY_OP(bfloat, log10_bf16, bfloat(log10(float(x))));
UNARY_OP(bfloat, log2_bf16, bfloat(log2(float(x))));
UNARY_OP(bfloat, exp_bf16, bfloat(exp(float(x))));
UNARY_OP(bfloat, exp10_bf16, bfloat(m_exp10(float(x))));
UNARY_OP(bfloat, exp2_bf16, bfloat(exp2(float(x))));
UNARY_OP(bfloat, softplus_bf16, bfloat(log(1.0f + exp(float(x)))));
UNARY_OP(bfloat, recip_bf16, bfloat(1.0f / float(x)));

UNARY_OP_OUTPUT(bfloat, bool, logical_not_bf16, float(x) == 0.0f);

// Operations with constants
UNARY_OP_WITH_CONSTANT(float, float, add_scalar_f32, x + const_val);
UNARY_OP_WITH_CONSTANT(float, float, sub_scalar_f32, x - const_val);
UNARY_OP_WITH_CONSTANT(float, float, mul_scalar_f32, x * const_val);
UNARY_OP_WITH_CONSTANT(float, float, div_scalar_f32, x / const_val);
UNARY_OP_WITH_CONSTANT(bool, bool, add_scalar_bool, x || const_val);
UNARY_OP_WITH_CONSTANT(bool, bool, sub_scalar_bool, x ^ const_val);
UNARY_OP_WITH_CONSTANT(bool, bool, mul_scalar_bool, x && const_val);
UNARY_OP_WITH_CONSTANT(bool, bool, div_scalar_bool, x && const_val);
UNARY_OP_WITH_CONSTANT(uint8_t, uint8_t, add_scalar_u8, x + const_val);
UNARY_OP_WITH_CONSTANT(uint8_t, uint8_t, sub_scalar_u8, (x > const_val) ? x - const_val : 0);
UNARY_OP_WITH_CONSTANT(uint8_t, uint8_t, mul_scalar_u8, x * const_val);
UNARY_OP_WITH_CONSTANT(uint8_t, uint8_t, div_scalar_u8, x / const_val);
UNARY_OP_WITH_CONSTANT(uint16_t, uint16_t, add_scalar_u16, x + const_val);
UNARY_OP_WITH_CONSTANT(uint16_t, uint16_t, sub_scalar_u16, (x > const_val) ? x - const_val : 0);
UNARY_OP_WITH_CONSTANT(uint16_t, uint16_t, mul_scalar_u16, x * const_val);
UNARY_OP_WITH_CONSTANT(uint16_t, uint16_t, div_scalar_u16, x / const_val);
UNARY_OP_WITH_CONSTANT(uint32_t, uint32_t, add_scalar_u32, x + const_val);
UNARY_OP_WITH_CONSTANT(uint32_t, uint32_t, sub_scalar_u32, (x > const_val) ? x - const_val : 0);
UNARY_OP_WITH_CONSTANT(uint32_t, uint32_t, mul_scalar_u32, x * const_val);
UNARY_OP_WITH_CONSTANT(uint32_t, uint32_t, div_scalar_u32, x / const_val);
UNARY_OP_WITH_CONSTANT(int8_t, int8_t, add_scalar_i8, x + const_val);
UNARY_OP_WITH_CONSTANT(int8_t, int8_t, sub_scalar_i8, x - const_val);
UNARY_OP_WITH_CONSTANT(int8_t, int8_t, mul_scalar_i8, x * const_val);
UNARY_OP_WITH_CONSTANT(int8_t, int8_t, div_scalar_i8, x / const_val);
UNARY_OP_WITH_CONSTANT(int16_t, int16_t, add_scalar_i16, x + const_val);
UNARY_OP_WITH_CONSTANT(int16_t, int16_t, sub_scalar_i16, x - const_val);
UNARY_OP_WITH_CONSTANT(int16_t, int16_t, mul_scalar_i16, x * const_val);
UNARY_OP_WITH_CONSTANT(int16_t, int16_t, div_scalar_i16, x / const_val);
UNARY_OP_WITH_CONSTANT(int32_t, int32_t, add_scalar_i32, x + const_val);
UNARY_OP_WITH_CONSTANT(int32_t, int32_t, sub_scalar_i32, x - const_val);
UNARY_OP_WITH_CONSTANT(int32_t, int32_t, mul_scalar_i32, x * const_val);
UNARY_OP_WITH_CONSTANT(int32_t, int32_t, div_scalar_i32, x / const_val);

// Maximum/minimum with const_val
UNARY_OP_WITH_CONSTANT(float, float, maximum_scalar_f32, maximum(x, const_val));
UNARY_OP_WITH_CONSTANT(bool, bool, maximum_scalar_bool, x || const_val);
UNARY_OP_WITH_CONSTANT(uint8_t, uint8_t, maximum_scalar_u8, maximum(x, const_val));
UNARY_OP_WITH_CONSTANT(uint16_t, uint16_t, maximum_scalar_u16, maximum(x, const_val));
UNARY_OP_WITH_CONSTANT(uint32_t, uint32_t, maximum_scalar_u32, maximum(x, const_val));
UNARY_OP_WITH_CONSTANT(int8_t, int8_t, maximum_scalar_i8, maximum(x, const_val));
UNARY_OP_WITH_CONSTANT(int16_t, int16_t, maximum_scalar_i16, maximum(x, const_val));
UNARY_OP_WITH_CONSTANT(int32_t, int32_t, maximum_scalar_i32, maximum(x, const_val));

UNARY_OP_WITH_CONSTANT(float, float, minimum_scalar_f32, minimum(x, const_val));
UNARY_OP_WITH_CONSTANT(bool, bool, minimum_scalar_bool, x && const_val);
UNARY_OP_WITH_CONSTANT(uint8_t, uint8_t, minimum_scalar_u8, minimum(x, const_val));
UNARY_OP_WITH_CONSTANT(uint16_t, uint16_t, minimum_scalar_u16, minimum(x, const_val));
UNARY_OP_WITH_CONSTANT(uint32_t, uint32_t, minimum_scalar_u32, minimum(x, const_val));
UNARY_OP_WITH_CONSTANT(int8_t, int8_t, minimum_scalar_i8, minimum(x, const_val));
UNARY_OP_WITH_CONSTANT(int16_t, int16_t, minimum_scalar_i16, minimum(x, const_val));
UNARY_OP_WITH_CONSTANT(int32_t, int32_t, minimum_scalar_i32, minimum(x, const_val));

// Power and activation functions with const_val
UNARY_OP_WITH_CONSTANT(float, float, pow_f32, m_pow_float(x, const_val));
UNARY_OP_WITH_CONSTANT(bool, bool, pow_bool, x && (const_val != 0));
UNARY_OP_WITH_CONSTANT(uint8_t, uint8_t, pow_u8, (uint8_t)m_pow_float((float)x, (float)const_val));
UNARY_OP_WITH_CONSTANT(uint16_t, uint16_t, pow_u16, (uint16_t)m_pow_float((float)x, (float)const_val));
UNARY_OP_WITH_CONSTANT(uint32_t, uint32_t, pow_u32, (uint32_t)m_pow_float((float)x, (float)const_val));
UNARY_OP_WITH_CONSTANT(int8_t, int8_t, pow_i8, (int8_t)m_pow_float((float)x, (float)const_val));
UNARY_OP_WITH_CONSTANT(int16_t, int16_t, pow_i16, (int16_t)m_pow_float((float)x, (float)const_val));
UNARY_OP_WITH_CONSTANT(int32_t, int32_t, pow_i32, (int32_t)m_pow_float((float)x, (float)const_val));

UNARY_OP_WITH_CONSTANT(float, float, leaky_relu_f32, x > 0 ? x : const_val * x);
UNARY_OP_WITH_CONSTANT(float, float, elu_f32, x > 0 ? x : const_val * (exp(x) - 1.0f));

// Comparison with const_val
UNARY_OP_WITH_CONSTANT(float, bool, eq_scalar_f32, x == const_val);
UNARY_OP_WITH_CONSTANT(float, bool, ne_scalar_f32, x != const_val);
UNARY_OP_WITH_CONSTANT(float, bool, lt_scalar_f32, x < const_val);
UNARY_OP_WITH_CONSTANT(float, bool, le_scalar_f32, x <= const_val);
UNARY_OP_WITH_CONSTANT(float, bool, gt_scalar_f32, x > const_val);
UNARY_OP_WITH_CONSTANT(float, bool, ge_scalar_f32, x >= const_val);
UNARY_OP_WITH_CONSTANT(bool, bool, eq_scalar_bool, x == const_val);
UNARY_OP_WITH_CONSTANT(bool, bool, ne_scalar_bool, x != const_val);
UNARY_OP_WITH_CONSTANT(bool, bool, lt_scalar_bool, !x && const_val);
UNARY_OP_WITH_CONSTANT(bool, bool, le_scalar_bool, !x || const_val);
UNARY_OP_WITH_CONSTANT(bool, bool, gt_scalar_bool, x && !const_val);
UNARY_OP_WITH_CONSTANT(bool, bool, ge_scalar_bool, x || !const_val);
UNARY_OP_WITH_CONSTANT(uint8_t, bool, eq_scalar_u8, x == const_val);
UNARY_OP_WITH_CONSTANT(uint8_t, bool, ne_scalar_u8, x != const_val);
UNARY_OP_WITH_CONSTANT(uint8_t, bool, lt_scalar_u8, x < const_val);
UNARY_OP_WITH_CONSTANT(uint8_t, bool, le_scalar_u8, x <= const_val);
UNARY_OP_WITH_CONSTANT(uint8_t, bool, gt_scalar_u8, x > const_val);
UNARY_OP_WITH_CONSTANT(uint8_t, bool, ge_scalar_u8, x >= const_val);
UNARY_OP_WITH_CONSTANT(uint16_t, bool, eq_scalar_u16, x == const_val);
UNARY_OP_WITH_CONSTANT(uint16_t, bool, ne_scalar_u16, x != const_val);
UNARY_OP_WITH_CONSTANT(uint16_t, bool, lt_scalar_u16, x < const_val);
UNARY_OP_WITH_CONSTANT(uint16_t, bool, le_scalar_u16, x <= const_val);
UNARY_OP_WITH_CONSTANT(uint16_t, bool, gt_scalar_u16, x > const_val);
UNARY_OP_WITH_CONSTANT(uint16_t, bool, ge_scalar_u16, x >= const_val);
UNARY_OP_WITH_CONSTANT(uint32_t, bool, eq_scalar_u32, x == const_val);
UNARY_OP_WITH_CONSTANT(uint32_t, bool, ne_scalar_u32, x != const_val);
UNARY_OP_WITH_CONSTANT(uint32_t, bool, lt_scalar_u32, x < const_val);
UNARY_OP_WITH_CONSTANT(uint32_t, bool, le_scalar_u32, x <= const_val);
UNARY_OP_WITH_CONSTANT(uint32_t, bool, gt_scalar_u32, x > const_val);
UNARY_OP_WITH_CONSTANT(uint32_t, bool, ge_scalar_u32, x >= const_val);
UNARY_OP_WITH_CONSTANT(int8_t, bool, eq_scalar_i8, x == const_val);
UNARY_OP_WITH_CONSTANT(int8_t, bool, ne_scalar_i8, x != const_val);
UNARY_OP_WITH_CONSTANT(int8_t, bool, lt_scalar_i8, x < const_val);
UNARY_OP_WITH_CONSTANT(int8_t, bool, le_scalar_i8, x <= const_val);
UNARY_OP_WITH_CONSTANT(int8_t, bool, gt_scalar_i8, x > const_val);
UNARY_OP_WITH_CONSTANT(int8_t, bool, ge_scalar_i8, x >= const_val);
UNARY_OP_WITH_CONSTANT(int16_t, bool, eq_scalar_i16, x == const_val);
UNARY_OP_WITH_CONSTANT(int16_t, bool, ne_scalar_i16, x != const_val);
UNARY_OP_WITH_CONSTANT(int16_t, bool, lt_scalar_i16, x < const_val);
UNARY_OP_WITH_CONSTANT(int16_t, bool, le_scalar_i16, x <= const_val);
UNARY_OP_WITH_CONSTANT(int16_t, bool, gt_scalar_i16, x > const_val);
UNARY_OP_WITH_CONSTANT(int16_t, bool, ge_scalar_i16, x >= const_val);
UNARY_OP_WITH_CONSTANT(int32_t, bool, eq_scalar_i32, x == const_val);
UNARY_OP_WITH_CONSTANT(int32_t, bool, ne_scalar_i32, x != const_val);
UNARY_OP_WITH_CONSTANT(int32_t, bool, lt_scalar_i32, x < const_val);
UNARY_OP_WITH_CONSTANT(int32_t, bool, le_scalar_i32, x <= const_val);
UNARY_OP_WITH_CONSTANT(int32_t, bool, gt_scalar_i32, x > const_val);
UNARY_OP_WITH_CONSTANT(int32_t, bool, ge_scalar_i32, x >= const_val);

// Half-precision floating point operations with const_val
UNARY_OP_WITH_CONSTANT(half, half, add_scalar_f16, x + const_val);
UNARY_OP_WITH_CONSTANT(half, half, sub_scalar_f16, x - const_val);
UNARY_OP_WITH_CONSTANT(half, half, mul_scalar_f16, x * const_val);
UNARY_OP_WITH_CONSTANT(half, half, div_scalar_f16, x / const_val);

// Improved half-precision maximum/minimum with better precision
UNARY_OP_WITH_CONSTANT(half, half, maximum_scalar_f16, half(maximum(float(x), float(const_val))));
UNARY_OP_WITH_CONSTANT(half, half, minimum_scalar_f16, half(minimum(float(x), float(const_val))));

UNARY_OP_WITH_CONSTANT(half, half, pow_f16, half(m_pow_float(float(x), float(const_val))));
UNARY_OP_WITH_CONSTANT(half, half, leaky_relu_f16, x > 0.0h ? x : half(float(const_val) * float(x)));
UNARY_OP_WITH_CONSTANT(half, half, elu_f16, half(float(x) > 0.0f ? float(x) : float(const_val) * (exp(float(x)) - 1.0f)));

// Improved half-precision comparison operations with better precision
UNARY_OP_WITH_CONSTANT(half, bool, eq_scalar_f16, float(x) == float(const_val));
UNARY_OP_WITH_CONSTANT(half, bool, ne_scalar_f16, float(x) != float(const_val));
UNARY_OP_WITH_CONSTANT(half, bool, lt_scalar_f16, float(x) < float(const_val));
UNARY_OP_WITH_CONSTANT(half, bool, le_scalar_f16, float(x) <= float(const_val));
UNARY_OP_WITH_CONSTANT(half, bool, gt_scalar_f16, float(x) > float(const_val));
UNARY_OP_WITH_CONSTANT(half, bool, ge_scalar_f16, float(x) >= float(const_val));

// BFloat16 operations with const_val
UNARY_OP_WITH_CONSTANT(bfloat, bfloat, add_scalar_bf16, x + const_val);
UNARY_OP_WITH_CONSTANT(bfloat, bfloat, sub_scalar_bf16, x - const_val);
UNARY_OP_WITH_CONSTANT(bfloat, bfloat, mul_scalar_bf16, x * const_val);
UNARY_OP_WITH_CONSTANT(bfloat, bfloat, div_scalar_bf16, x / const_val);
UNARY_OP_WITH_CONSTANT(bfloat, bfloat, maximum_scalar_bf16, bfloat(maximum(float(x), float(const_val))));
UNARY_OP_WITH_CONSTANT(bfloat, bfloat, minimum_scalar_bf16, bfloat(minimum(float(x), float(const_val))));
UNARY_OP_WITH_CONSTANT(bfloat, bfloat, pow_bf16, bfloat(m_pow_float(float(x), float(const_val))));
UNARY_OP_WITH_CONSTANT(bfloat, bfloat, leaky_relu_bf16, x > 0.0bf ? x : bfloat(float(const_val) * float(x)));
UNARY_OP_WITH_CONSTANT(bfloat, bfloat, elu_bf16, bfloat(float(x) > 0.0f ? float(x) : float(const_val) * (exp(float(x)) - 1.0f)));

UNARY_OP_WITH_CONSTANT(bfloat, bool, eq_scalar_bf16, float(x) == float(const_val));
UNARY_OP_WITH_CONSTANT(bfloat, bool, ne_scalar_bf16, float(x) != float(const_val));
UNARY_OP_WITH_CONSTANT(bfloat, bool, lt_scalar_bf16, float(x) < float(const_val));
UNARY_OP_WITH_CONSTANT(bfloat, bool, le_scalar_bf16, float(x) <= float(const_val));
UNARY_OP_WITH_CONSTANT(bfloat, bool, gt_scalar_bf16, float(x) > float(const_val));
UNARY_OP_WITH_CONSTANT(bfloat, bool, ge_scalar_bf16, float(x) >= float(const_val));
