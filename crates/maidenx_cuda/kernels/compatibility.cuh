#include <cstdint>

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
