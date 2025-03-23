#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <cstdint>

__device__ __forceinline__ uint8_t atomicAdd(uint8_t *address, uint8_t val) {
  unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
  unsigned int shift = (((size_t)address & 3) * 8);
  unsigned int old, assumed, sum, new_val;

  old = *base_address;
  do {
    assumed = old;
    sum = val + ((assumed >> shift) & 0xFF);
    new_val = (assumed & ~(0xFF << shift)) | ((sum & 0xFF) << shift);
    old = atomicCAS(base_address, assumed, new_val);
  } while (assumed != old);
  return (old >> shift) & 0xFF;
}

__device__ __forceinline__ uint16_t atomicAdd(uint16_t *address, uint16_t val) {
  unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
  unsigned int shift = (((size_t)address & 3) * 8);
  unsigned int old, assumed, sum, new_val;
  old = *base_address;
  do {
    assumed = old;
    sum = val + ((assumed >> shift) & 0xFFFF);
    new_val = (assumed & ~(0xFFFF << shift)) | ((sum & 0xFFFF) << shift);
    old = atomicCAS(base_address, assumed, new_val);
  } while (assumed != old);
  return (old >> shift) & 0xFFFF;
}

__device__ __forceinline__ uint64_t atomicAdd(uint64_t *address, uint64_t val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, assumed + val);
  } while (assumed != old);
  return old;
}

__device__ __forceinline__ int8_t atomicAdd(int8_t *address, int8_t val) {
  unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
  unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4321};
  unsigned int shift = (((size_t)address & 3) * 8);
  unsigned int selector = selectors[(size_t)address & 3];
  unsigned int old, assumed, sum, new_val;

  old = *base_address;
  do {
    assumed = old;
    sum = static_cast<unsigned int>(val) +
          static_cast<unsigned int>((assumed >> shift) & 0xFF);
    new_val = (assumed & ~(0xFF << shift)) | ((sum & 0xFF) << shift);
    old = atomicCAS(base_address, assumed, new_val);
  } while (assumed != old);
  return (old >> shift) & 0xFF;
}

__device__ __forceinline__ int16_t atomicAdd(int16_t *address, int16_t val) {
  unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
  unsigned int shift = (((size_t)address & 3) * 8);
  unsigned int old, assumed, sum, new_val;
  old = *base_address;
  do {
    assumed = old;
    sum = static_cast<unsigned int>(val) +
          static_cast<unsigned int>((assumed >> shift) & 0xFFFF);
    new_val = (assumed & ~(0xFFFF << shift)) | ((sum & 0xFFFF) << shift);
    old = atomicCAS(base_address, assumed, new_val);
  } while (assumed != old);
  return (old >> shift) & 0xFFFF;
}

__device__ __forceinline__ int64_t atomicAdd(int64_t *address, int64_t val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    static_cast<unsigned long long int>(
                        static_cast<int64_t>(assumed) + val));
  } while (assumed != old);
  return static_cast<int64_t>(old);
}

__device__ __forceinline__ uint8_t atomicMax(uint8_t *address, uint8_t val) {
  unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
  unsigned int shift = (((size_t)address & 3) * 8);
  unsigned int old, assumed, new_val;
  old = *base_address;
  do {
    assumed = old;
    uint8_t current = (assumed >> shift) & 0xFF;
    uint8_t max_val = current > val ? current : val;
    new_val = (assumed & ~(0xFF << shift)) | ((max_val & 0xFF) << shift);
    old = atomicCAS(base_address, assumed, new_val);
  } while (assumed != old);
  return (old >> shift) & 0xFF;
}

__device__ __forceinline__ uint16_t atomicMax(uint16_t *address, uint16_t val) {
  unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
  unsigned int shift = (((size_t)address & 3) * 8);
  unsigned int old, assumed, new_val;
  old = *base_address;
  do {
    assumed = old;
    uint16_t current = (assumed >> shift) & 0xFFFF;
    uint16_t max_val = current > val ? current : val;
    new_val = (assumed & ~(0xFFFF << shift)) | ((max_val & 0xFFFF) << shift);
    old = atomicCAS(base_address, assumed, new_val);
  } while (assumed != old);
  return (old >> shift) & 0xFFFF;
}

__device__ __forceinline__ uint64_t atomicMax(uint64_t *address, uint64_t val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;
  do {
    assumed = old;
    uint64_t current = static_cast<uint64_t>(assumed);
    uint64_t max_val = current > val ? current : val;
    old = atomicCAS(address_as_ull, assumed,
                    static_cast<unsigned long long int>(max_val));
  } while (assumed != old);
  return static_cast<uint64_t>(old);
}

__device__ __forceinline__ int8_t atomicMax(int8_t *address, int8_t val) {
  unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
  unsigned int shift = (((size_t)address & 3) * 8);
  unsigned int old, assumed, new_val;
  old = *base_address;
  do {
    assumed = old;
    int8_t current = (assumed >> shift) & 0xFF;
    int8_t max_val = current > val ? current : val;
    new_val = (assumed & ~(0xFF << shift)) | ((max_val & 0xFF) << shift);
    old = atomicCAS(base_address, assumed, new_val);
  } while (assumed != old);
  return (old >> shift) & 0xFF;
}

__device__ __forceinline__ int16_t atomicMax(int16_t *address, int16_t val) {
  unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
  unsigned int shift = (((size_t)address & 3) * 8);
  unsigned int old, assumed, new_val;
  old = *base_address;
  do {
    assumed = old;
    int16_t current = (assumed >> shift) & 0xFFFF;
    int16_t max_val = current > val ? current : val;
    new_val = (assumed & ~(0xFFFF << shift)) | ((max_val & 0xFFFF) << shift);
    old = atomicCAS(base_address, assumed, new_val);
  } while (assumed != old);
  return (old >> shift) & 0xFFFF;
}

__device__ __forceinline__ int64_t atomicMax(int64_t *address, int64_t val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;
  do {
    assumed = old;
    int64_t current = static_cast<int64_t>(assumed);
    int64_t max_val = current > val ? current : val;
    old = atomicCAS(address_as_ull, assumed,
                    static_cast<unsigned long long int>(max_val));
  } while (assumed != old);
  return static_cast<int64_t>(old);
}

__device__ __forceinline__ float atomicMax(float *address, float val) {
  int *address_as_int = (int *)address;
  int old = *address_as_int;
  int assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_int, assumed,
                    __float_as_int(fmaxf(__int_as_float(assumed), val)));
  } while (assumed != old);
  return __int_as_float(old);
}

__device__ __forceinline__ double atomicMax(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;
  do {
    assumed = old;
    old = atomicCAS(
        address_as_ull, assumed,
        __double_as_longlong(fmax(__longlong_as_double(assumed), val)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

__device__ __forceinline__ __half atomicMax(__half *address, __half val) {
  unsigned short int *address_as_ushort = (unsigned short int *)address;
  unsigned short int old = *address_as_ushort;
  unsigned short int assumed;
  do {
    assumed = old;
    __half current = __ushort_as_half(assumed);
    __half max_val = __hgt(current, val) ? current : val;
    old = atomicCAS(address_as_ushort, assumed, __half_as_ushort(max_val));
  } while (assumed != old);
  return __ushort_as_half(old);
}

__device__ __forceinline__ __nv_bfloat16 atomicMax(__nv_bfloat16 *address,
                                                   __nv_bfloat16 val) {
  unsigned short int *address_as_ushort = (unsigned short int *)address;
  unsigned short int old = *address_as_ushort;
  unsigned short int assumed;
  do {
    assumed = old;
    __nv_bfloat16 current = __ushort_as_bfloat16(assumed);
    __nv_bfloat16 max_val = __hgt(current, val) ? current : val;
    old = atomicCAS(address_as_ushort, assumed, __bfloat16_as_ushort(max_val));
  } while (assumed != old);
  return __ushort_as_bfloat16(old);
}

__device__ __forceinline__ uint8_t atomicMin(uint8_t *address, uint8_t val) {
  unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
  unsigned int shift = (((size_t)address & 3) * 8);
  unsigned int old, assumed, new_val;
  old = *base_address;
  do {
    assumed = old;
    uint8_t current = (assumed >> shift) & 0xFF;
    uint8_t min_val = current < val ? current : val;
    new_val = (assumed & ~(0xFF << shift)) | ((min_val & 0xFF) << shift);
    old = atomicCAS(base_address, assumed, new_val);
  } while (assumed != old);
  return (old >> shift) & 0xFF;
}

__device__ __forceinline__ uint16_t atomicMin(uint16_t *address, uint16_t val) {
  unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
  unsigned int shift = (((size_t)address & 3) * 8);
  unsigned int old, assumed, new_val;
  old = *base_address;
  do {
    assumed = old;
    uint16_t current = (assumed >> shift) & 0xFFFF;
    uint16_t min_val = current < val ? current : val;
    new_val = (assumed & ~(0xFFFF << shift)) | ((min_val & 0xFFFF) << shift);
    old = atomicCAS(base_address, assumed, new_val);
  } while (assumed != old);
  return (old >> shift) & 0xFFFF;
}

__device__ __forceinline__ uint64_t atomicMin(uint64_t *address, uint64_t val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;
  do {
    assumed = old;
    uint64_t current = static_cast<uint64_t>(assumed);
    uint64_t min_val = current < val ? current : val;
    old = atomicCAS(address_as_ull, assumed,
                    static_cast<unsigned long long int>(min_val));
  } while (assumed != old);
  return static_cast<uint64_t>(old);
}

__device__ __forceinline__ int8_t atomicMin(int8_t *address, int8_t val) {
  unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
  unsigned int shift = (((size_t)address & 3) * 8);
  unsigned int old, assumed, new_val;
  old = *base_address;
  do {
    assumed = old;
    int8_t current = (assumed >> shift) & 0xFF;
    int8_t min_val = current < val ? current : val;
    new_val = (assumed & ~(0xFF << shift)) | ((min_val & 0xFF) << shift);
    old = atomicCAS(base_address, assumed, new_val);
  } while (assumed != old);
  return (old >> shift) & 0xFF;
}

__device__ __forceinline__ int16_t atomicMin(int16_t *address, int16_t val) {
  unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
  unsigned int shift = (((size_t)address & 3) * 8);
  unsigned int old, assumed, new_val;
  old = *base_address;
  do {
    assumed = old;
    int16_t current = (assumed >> shift) & 0xFFFF;
    int16_t min_val = current < val ? current : val;
    new_val = (assumed & ~(0xFFFF << shift)) | ((min_val & 0xFFFF) << shift);
    old = atomicCAS(base_address, assumed, new_val);
  } while (assumed != old);
  return (old >> shift) & 0xFFFF;
}

__device__ __forceinline__ int64_t atomicMin(int64_t *address, int64_t val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;
  do {
    assumed = old;
    int64_t current = static_cast<int64_t>(assumed);
    int64_t min_val = current < val ? current : val;
    old = atomicCAS(address_as_ull, assumed,
                    static_cast<unsigned long long int>(min_val));
  } while (assumed != old);
  return static_cast<int64_t>(old);
}

__device__ __forceinline__ float atomicMin(float *address, float val) {
  int *address_as_int = (int *)address;
  int old = *address_as_int;
  int assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_int, assumed,
                    __float_as_int(fminf(__int_as_float(assumed), val)));
  } while (assumed != old);
  return __int_as_float(old);
}

__device__ __forceinline__ double atomicMin(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;
  do {
    assumed = old;
    old = atomicCAS(
        address_as_ull, assumed,
        __double_as_longlong(fmin(__longlong_as_double(assumed), val)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

__device__ __forceinline__ __half atomicMin(__half *address, __half val) {
  unsigned short int *address_as_ushort = (unsigned short int *)address;
  unsigned short int old = *address_as_ushort;
  unsigned short int assumed;
  do {
    assumed = old;
    __half current = __ushort_as_half(assumed);
    __half min_val = __hlt(current, val) ? current : val;
    old = atomicCAS(address_as_ushort, assumed, __half_as_ushort(min_val));
  } while (assumed != old);
  return __ushort_as_half(old);
}

__device__ __forceinline__ __nv_bfloat16 atomicMin(__nv_bfloat16 *address,
                                                   __nv_bfloat16 val) {
  unsigned short int *address_as_ushort = (unsigned short int *)address;
  unsigned short int old = *address_as_ushort;
  unsigned short int assumed;
  do {
    assumed = old;
    __nv_bfloat16 current = __ushort_as_bfloat16(assumed);
    __nv_bfloat16 min_val = __hlt(current, val) ? current : val;
    old = atomicCAS(address_as_ushort, assumed, __bfloat16_as_ushort(min_val));
  } while (assumed != old);
  return __ushort_as_bfloat16(old);
}
