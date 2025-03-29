#include <metal_stdlib>
#include <metal_atomic>

using namespace metal;

// For uint8_t
inline uint8_t atomic_add_uint8(device uint8_t* address, uint8_t val) {
    device atomic_uint* base_address = (device atomic_uint*)((device uintptr_t)(address) & ~3);
    uint shift = (((device uintptr_t)(address) & 3) * 8);
    uint mask = 0xFF << shift;
    
    uint old = atomic_load_explicit(base_address, memory_order_relaxed);
    uint assumed, sum, new_val;
    
    do {
        assumed = old;
        uint8_t current = (assumed >> shift) & 0xFF;
        sum = val + current;
        new_val = (assumed & ~mask) | ((sum & 0xFF) << shift);
        old = atomic_exchange_explicit(base_address, new_val, memory_order_relaxed);
    } while (assumed != old);
    
    return (old >> shift) & 0xFF;
}

// For int8_t
inline int8_t atomic_add_int8(device int8_t* address, int8_t val) {
    device atomic_uint* base_address = (device atomic_uint*)((device uintptr_t)(address) & ~3);
    uint shift = (((device uintptr_t)(address) & 3) * 8);
    uint mask = 0xFF << shift;
    
    uint old = atomic_load_explicit(base_address, memory_order_relaxed);
    uint assumed, sum, new_val;
    
    do {
        assumed = old;
        int8_t current = (assumed >> shift) & 0xFF;
        sum = val + current;
        new_val = (assumed & ~mask) | ((sum & 0xFF) << shift);
        old = atomic_exchange_explicit(base_address, new_val, memory_order_relaxed);
    } while (assumed != old);
    
    return (old >> shift) & 0xFF;
}

// For uint16_t
inline uint16_t atomic_add_uint16(device uint16_t* address, uint16_t val) {
    if ((device uintptr_t)(address) % 2 == 0) {
        // Aligned case - can use atomic_fetch_add directly
        return atomic_fetch_add_explicit((device atomic_uint*)(address), val, memory_order_relaxed);
    } else {
        // Unaligned case
        device atomic_uint* base_address = (device atomic_uint*)((device uintptr_t)(address) & ~3);
        uint shift = (((device uintptr_t)(address) & 3) * 8);
        uint mask = 0xFFFF << shift;
        
        uint old = atomic_load_explicit(base_address, memory_order_relaxed);
        uint assumed, sum, new_val;
        
        do {
            assumed = old;
            uint16_t current = (assumed >> shift) & 0xFFFF;
            sum = val + current;
            new_val = (assumed & ~mask) | ((sum & 0xFFFF) << shift);
            old = atomic_exchange_explicit(base_address, new_val, memory_order_relaxed);
        } while (assumed != old);
        
        return (old >> shift) & 0xFFFF;
    }
}

// For int16_t
inline int16_t atomic_add_int16(device int16_t* address, int16_t val) {
    if ((device uintptr_t)(address) % 2 == 0) {
        // Aligned case - can use atomic_fetch_add directly
        return atomic_fetch_add_explicit((device atomic_int*)(address), val, memory_order_relaxed);
    } else {
        // Unaligned case
        device atomic_uint* base_address = (device atomic_uint*)((device uintptr_t)(address) & ~3);
        uint shift = (((device uintptr_t)(address) & 3) * 8);
        uint mask = 0xFFFF << shift;
        
        uint old = atomic_load_explicit(base_address, memory_order_relaxed);
        uint assumed, sum, new_val;
        
        do {
            assumed = old;
            int16_t current = (assumed >> shift) & 0xFFFF;
            sum = val + current;
            new_val = (assumed & ~mask) | ((sum & 0xFFFF) << shift);
            old = atomic_exchange_explicit(base_address, new_val, memory_order_relaxed);
        } while (assumed != old);
        
        return (old >> shift) & 0xFFFF;
    }
}

// For float - using atomic_uint for bit manipulation
inline float atomic_add_float(device float* address, float val) {
    device atomic_uint* address_as_uint = (device atomic_uint*)(address);
    uint old_val_uint = atomic_load_explicit(address_as_uint, memory_order_relaxed);
    float old_val = *((thread float*)&old_val_uint);
    float new_val;
    uint new_val_uint;
    
    do {
        new_val = old_val + val;
        new_val_uint = *((thread uint*)&new_val);
        bool success = atomic_compare_exchange_weak_explicit(
            address_as_uint,
            &old_val_uint,
            new_val_uint,
            memory_order_relaxed,
            memory_order_relaxed
        );
        
        if (success) {
            return old_val;
        }
        
        old_val = *((thread float*)&old_val_uint);
    } while (true);
}

inline uint32_t atomic_add_uint32(device uint32_t* address, uint32_t val) {
    return atomic_fetch_add_explicit((device atomic_uint*)(address), val, memory_order_relaxed);
}

inline int32_t atomic_add_int32(device int32_t* address, int32_t val) {
    return atomic_fetch_add_explicit((device atomic_int*)(address), val, memory_order_relaxed);
}

// For half
inline half atomic_add_half(device half* address, half val) {
    // Use uint16_t instead of atomic_ushort
    device atomic_uint* base_address = (device atomic_uint*)((device uintptr_t)(address) & ~3);
    uint shift = (((device uintptr_t)(address) & 2) * 8); // Shift by 0 or 16 bits
    uint mask = 0xFFFF << shift;
    
    uint old = atomic_load_explicit(base_address, memory_order_relaxed);
    uint assumed, new_val;
    ushort old_val_ushort = (old >> shift) & 0xFFFF;
    half old_val = *((thread half*)&old_val_ushort);
    
    do {
        assumed = old;
        half new_half_val = old_val + val;
        ushort new_half_ushort = *((thread ushort*)&new_half_val);
        new_val = (assumed & ~mask) | ((uint)new_half_ushort << shift);
        old = atomic_exchange_explicit(base_address, new_val, memory_order_relaxed);
        
        if (assumed == old) {
            return old_val;
        }
        
        old_val_ushort = (old >> shift) & 0xFFFF;
        old_val = *((thread half*)&old_val_ushort);
    } while (true);
}

// For bfloat
inline bfloat atomic_add_bfloat(device bfloat* address, bfloat val) {
    // Use uint16_t instead of atomic_ushort
    device atomic_uint* base_address = (device atomic_uint*)((device uintptr_t)(address) & ~3);
    uint shift = (((device uintptr_t)(address) & 2) * 8); // Shift by 0 or 16 bits
    uint mask = 0xFFFF << shift;
    
    uint old = atomic_load_explicit(base_address, memory_order_relaxed);
    uint assumed, new_val;
    ushort old_val_ushort = (old >> shift) & 0xFFFF;
    bfloat old_val = *((thread bfloat*)&old_val_ushort);
    
    do {
        assumed = old;
        bfloat new_bfloat_val = old_val + val;
        ushort new_bfloat_ushort = *((thread ushort*)&new_bfloat_val);
        new_val = (assumed & ~mask) | ((uint)new_bfloat_ushort << shift);
        old = atomic_exchange_explicit(base_address, new_val, memory_order_relaxed);
        
        if (assumed == old) {
            return old_val;
        }
        
        old_val_ushort = (old >> shift) & 0xFFFF;
        old_val = *((thread bfloat*)&old_val_ushort);
    } while (true);
}

// ==== Atomic Max Operations ====

// For float
inline float atomic_max_float(device float* address, float val) {
    device atomic_uint* address_as_uint = (device atomic_uint*)(address);
    uint old_val_uint = atomic_load_explicit(address_as_uint, memory_order_relaxed);
    float old_val = *((thread float*)&old_val_uint);
    
    if (old_val >= val) {
        return old_val; // No update needed
    }
    
    do {
        float new_val = max(old_val, val);
        uint new_val_uint = *((thread uint*)&new_val);
        bool success = atomic_compare_exchange_weak_explicit(
            address_as_uint,
            &old_val_uint,
            new_val_uint,
            memory_order_relaxed,
            memory_order_relaxed
        );
        
        if (success) {
            return old_val;
        }
        
        old_val = *((thread float*)&old_val_uint);
        if (old_val >= val) {
            return old_val; // Someone else updated with a larger value
        }
    } while (true);
}

// For uint8_t
inline uint8_t atomic_max_uint8(device uint8_t* address, uint8_t val) {
    device atomic_uint* base_address = (device atomic_uint*)((device uintptr_t)(address) & ~3);
    uint shift = (((device uintptr_t)(address) & 3) * 8);
    uint mask = 0xFF << shift;
    
    uint old = atomic_load_explicit(base_address, memory_order_relaxed);
    uint assumed, new_val;
    
    do {
        assumed = old;
        uint8_t current = (assumed >> shift) & 0xFF;
        
        if (current >= val) {
            return current; // No update needed
        }
        
        new_val = (assumed & ~mask) | ((val & 0xFF) << shift);
        old = atomic_exchange_explicit(base_address, new_val, memory_order_relaxed);
    } while (assumed != old);
    
    return (old >> shift) & 0xFF;
}

// For int8_t
inline int8_t atomic_max_int8(device int8_t* address, int8_t val) {
    device atomic_uint* base_address = (device atomic_uint*)((device uintptr_t)(address) & ~3);
    uint shift = (((device uintptr_t)(address) & 3) * 8);
    uint mask = 0xFF << shift;
    
    uint old = atomic_load_explicit(base_address, memory_order_relaxed);
    uint assumed, new_val;
    
    do {
        assumed = old;
        int8_t current = (assumed >> shift) & 0xFF;
        
        if (current >= val) {
            return current; // No update needed
        }
        
        new_val = (assumed & ~mask) | ((val & 0xFF) << shift);
        old = atomic_exchange_explicit(base_address, new_val, memory_order_relaxed);
    } while (assumed != old);
    
    return (old >> shift) & 0xFF;
}

// For uint16_t
inline uint16_t atomic_max_uint16(device uint16_t* address, uint16_t val) {
    if ((device uintptr_t)(address) % 2 == 0) {
        // Aligned case
        return atomic_fetch_max_explicit((device atomic_uint*)(address), val, memory_order_relaxed);
    } else {
        // Unaligned case
        device atomic_uint* base_address = (device atomic_uint*)((device uintptr_t)(address) & ~3);
        uint shift = (((device uintptr_t)(address) & 3) * 8);
        uint mask = 0xFFFF << shift;
        
        uint old = atomic_load_explicit(base_address, memory_order_relaxed);
        uint assumed, new_val;
        
        do {
            assumed = old;
            uint16_t current = (assumed >> shift) & 0xFFFF;
            
            if (current >= val) {
                return current; // No update needed
            }
            
            new_val = (assumed & ~mask) | ((val & 0xFFFF) << shift);
            old = atomic_exchange_explicit(base_address, new_val, memory_order_relaxed);
        } while (assumed != old);
        
        return (old >> shift) & 0xFFFF;
    }
}

// For int16_t
inline int16_t atomic_max_int16(device int16_t* address, int16_t val) {
    if ((device uintptr_t)(address) % 2 == 0) {
        // Aligned case
        return atomic_fetch_max_explicit((device atomic_int*)(address), val, memory_order_relaxed);
    } else {
        // Unaligned case
        device atomic_uint* base_address = (device atomic_uint*)((device uintptr_t)(address) & ~3);
        uint shift = (((device uintptr_t)(address) & 3) * 8);
        uint mask = 0xFFFF << shift;
        
        uint old = atomic_load_explicit(base_address, memory_order_relaxed);
        uint assumed, new_val;
        
        do {
            assumed = old;
            int16_t current = (assumed >> shift) & 0xFFFF;
            
            if (current >= val) {
                return current; // No update needed
            }
            
            new_val = (assumed & ~mask) | ((val & 0xFFFF) << shift);
            old = atomic_exchange_explicit(base_address, new_val, memory_order_relaxed);
        } while (assumed != old);
        
        return (old >> shift) & 0xFFFF;
    }
}

inline uint32_t atomic_max_uint32(device uint32_t* address, uint32_t val) {
    return atomic_fetch_max_explicit((device atomic_uint*)(address), val, memory_order_relaxed);
}

inline int32_t atomic_max_int32(device int32_t* address, int32_t val) {
    return atomic_fetch_max_explicit((device atomic_int*)(address), val, memory_order_relaxed);
}

// For half
inline half atomic_max_half(device half* address, half val) {
    // Use uint16_t instead of atomic_ushort
    device atomic_uint* base_address = (device atomic_uint*)((device uintptr_t)(address) & ~3);
    uint shift = (((device uintptr_t)(address) & 2) * 8); // Shift by 0 or 16 bits
    uint mask = 0xFFFF << shift;
    
    uint old = atomic_load_explicit(base_address, memory_order_relaxed);
    uint assumed, new_val;
    ushort old_val_ushort = (old >> shift) & 0xFFFF;
    half old_val = *((thread half*)&old_val_ushort);
    
    if (old_val >= val) {
        return old_val; // No update needed
    }
    
    do {
        assumed = old;
        // Using explicit cast to avoid ambiguity
        half new_half_val = (old_val >= val) ? old_val : val;
        ushort new_half_ushort = *((thread ushort*)&new_half_val);
        new_val = (assumed & ~mask) | ((uint)new_half_ushort << shift);
        old = atomic_exchange_explicit(base_address, new_val, memory_order_relaxed);
        
        if (assumed == old) {
            return old_val;
        }
        
        old_val_ushort = (old >> shift) & 0xFFFF;
        old_val = *((thread half*)&old_val_ushort);
        if (old_val >= val) {
            return old_val; // Someone else updated with a larger value
        }
    } while (true);
}

// For bfloat
inline bfloat atomic_max_bfloat(device bfloat* address, bfloat val) {
    // Use uint16_t instead of atomic_ushort
    device atomic_uint* base_address = (device atomic_uint*)((device uintptr_t)(address) & ~3);
    uint shift = (((device uintptr_t)(address) & 2) * 8); // Shift by 0 or 16 bits
    uint mask = 0xFFFF << shift;
    
    uint old = atomic_load_explicit(base_address, memory_order_relaxed);
    uint assumed, new_val;
    ushort old_val_ushort = (old >> shift) & 0xFFFF;
    bfloat old_val = *((thread bfloat*)&old_val_ushort);
    
    if (old_val >= val) {
        return old_val; // No update needed
    }
    
    do {
        assumed = old;
        // Using explicit comparison instead of max() to avoid ambiguity
        bfloat new_bfloat_val = (old_val >= val) ? old_val : val;
        ushort new_bfloat_ushort = *((thread ushort*)&new_bfloat_val);
        new_val = (assumed & ~mask) | ((uint)new_bfloat_ushort << shift);
        old = atomic_exchange_explicit(base_address, new_val, memory_order_relaxed);
        
        if (assumed == old) {
            return old_val;
        }
        
        old_val_ushort = (old >> shift) & 0xFFFF;
        old_val = *((thread bfloat*)&old_val_ushort);
        if (old_val >= val) {
            return old_val; // Someone else updated with a larger value
        }
    } while (true);
}

// ==== Atomic Min Operations ====

// For float
inline float atomic_min_float(device float* address, float val) {
    device atomic_uint* address_as_uint = (device atomic_uint*)(address);
    uint old_val_uint = atomic_load_explicit(address_as_uint, memory_order_relaxed);
    float old_val = *((thread float*)&old_val_uint);
    
    if (old_val <= val) {
        return old_val; // No update needed
    }
    
    do {
        float new_val = min(old_val, val);
        uint new_val_uint = *((thread uint*)&new_val);
        bool success = atomic_compare_exchange_weak_explicit(
            address_as_uint,
            &old_val_uint,
            new_val_uint,
            memory_order_relaxed,
            memory_order_relaxed
        );
        
        if (success) {
            return old_val;
        }
        
        old_val = *((thread float*)&old_val_uint);
        if (old_val <= val) {
            return old_val; // Someone else updated with a smaller value
        }
    } while (true);
}

// For uint8_t
inline uint8_t atomic_min_uint8(device uint8_t* address, uint8_t val) {
    device atomic_uint* base_address = (device atomic_uint*)((device uintptr_t)(address) & ~3);
    uint shift = (((device uintptr_t)(address) & 3) * 8);
    uint mask = 0xFF << shift;
    
    uint old = atomic_load_explicit(base_address, memory_order_relaxed);
    uint assumed, new_val;
    
    do {
        assumed = old;
        uint8_t current = (assumed >> shift) & 0xFF;
        
        if (current <= val) {
            return current; // No update needed
        }
        
        new_val = (assumed & ~mask) | ((val & 0xFF) << shift);
        old = atomic_exchange_explicit(base_address, new_val, memory_order_relaxed);
    } while (assumed != old);
    
    return (old >> shift) & 0xFF;
}

// For int8_t
inline int8_t atomic_min_int8(device int8_t* address, int8_t val) {
    device atomic_uint* base_address = (device atomic_uint*)((device uintptr_t)(address) & ~3);
    uint shift = (((device uintptr_t)(address) & 3) * 8);
    uint mask = 0xFF << shift;
    
    uint old = atomic_load_explicit(base_address, memory_order_relaxed);
    uint assumed, new_val;
    
    do {
        assumed = old;
        int8_t current = (assumed >> shift) & 0xFF;
        
        if (current <= val) {
            return current; // No update needed
        }
        
        new_val = (assumed & ~mask) | ((val & 0xFF) << shift);
        old = atomic_exchange_explicit(base_address, new_val, memory_order_relaxed);
    } while (assumed != old);
    
    return (old >> shift) & 0xFF;
}

// For uint16_t
inline uint16_t atomic_min_uint16(device uint16_t* address, uint16_t val) {
    if ((device uintptr_t)(address) % 2 == 0) {
        // Aligned case
        return atomic_fetch_min_explicit((device atomic_uint*)(address), val, memory_order_relaxed);
    } else {
        // Unaligned case
        device atomic_uint* base_address = (device atomic_uint*)((device uintptr_t)(address) & ~3);
        uint shift = (((device uintptr_t)(address) & 3) * 8);
        uint mask = 0xFFFF << shift;
        
        uint old = atomic_load_explicit(base_address, memory_order_relaxed);
        uint assumed, new_val;
        
        do {
            assumed = old;
            uint16_t current = (assumed >> shift) & 0xFFFF;
            
            if (current <= val) {
                return current; // No update needed
            }
            
            new_val = (assumed & ~mask) | ((val & 0xFFFF) << shift);
            old = atomic_exchange_explicit(base_address, new_val, memory_order_relaxed);
        } while (assumed != old);
        
        return (old >> shift) & 0xFFFF;
    }
}

// For int16_t
inline int16_t atomic_min_int16(device int16_t* address, int16_t val) {
    if ((device uintptr_t)(address) % 2 == 0) {
        // Aligned case
        return atomic_fetch_min_explicit((device atomic_int*)(address), val, memory_order_relaxed);
    } else {
        // Unaligned case
        device atomic_uint* base_address = (device atomic_uint*)((device uintptr_t)(address) & ~3);
        uint shift = (((device uintptr_t)(address) & 3) * 8);
        uint mask = 0xFFFF << shift;
        
        uint old = atomic_load_explicit(base_address, memory_order_relaxed);
        uint assumed, new_val;
        
        do {
            assumed = old;
            int16_t current = (assumed >> shift) & 0xFFFF;
            
            if (current <= val) {
                return current; // No update needed
            }
            
            new_val = (assumed & ~mask) | ((val & 0xFFFF) << shift);
            old = atomic_exchange_explicit(base_address, new_val, memory_order_relaxed);
        } while (assumed != old);
        
        return (old >> shift) & 0xFFFF;
    }
}

inline uint32_t atomic_min_uint32(device uint32_t* address, uint32_t val) {
    return atomic_fetch_min_explicit((device atomic_uint*)(address), val, memory_order_relaxed);
}

inline int32_t atomic_min_int32(device int32_t* address, int32_t val) {
    return atomic_fetch_min_explicit((device atomic_int*)(address), val, memory_order_relaxed);
}

// For half
inline half atomic_min_half(device half* address, half val) {
    // Use uint16_t instead of atomic_ushort
    device atomic_uint* base_address = (device atomic_uint*)((device uintptr_t)(address) & ~3);
    uint shift = (((device uintptr_t)(address) & 2) * 8); // Shift by 0 or 16 bits
    uint mask = 0xFFFF << shift;
    
    uint old = atomic_load_explicit(base_address, memory_order_relaxed);
    uint assumed, new_val;
    ushort old_val_ushort = (old >> shift) & 0xFFFF;
    half old_val = *((thread half*)&old_val_ushort);
    
    if (old_val <= val) {
        return old_val; // No update needed
    }
    
    do {
        assumed = old;
        // Using explicit comparison instead of min() to avoid ambiguity
        half new_half_val = (old_val <= val) ? old_val : val;
        ushort new_half_ushort = *((thread ushort*)&new_half_val);
        new_val = (assumed & ~mask) | ((uint)new_half_ushort << shift);
        old = atomic_exchange_explicit(base_address, new_val, memory_order_relaxed);
        
        if (assumed == old) {
            return old_val;
        }
        
        old_val_ushort = (old >> shift) & 0xFFFF;
        old_val = *((thread half*)&old_val_ushort);
        if (old_val <= val) {
            return old_val; // Someone else updated with a smaller value
        }
    } while (true);
}

// For bfloat
inline bfloat atomic_min_bfloat(device bfloat* address, bfloat val) {
    // Use uint16_t instead of atomic_ushort
    device atomic_uint* base_address = (device atomic_uint*)((device uintptr_t)(address) & ~3);
    uint shift = (((device uintptr_t)(address) & 2) * 8); // Shift by 0 or 16 bits
    uint mask = 0xFFFF << shift;
    
    uint old = atomic_load_explicit(base_address, memory_order_relaxed);
    uint assumed, new_val;
    ushort old_val_ushort = (old >> shift) & 0xFFFF;
    bfloat old_val = *((thread bfloat*)&old_val_ushort);
    
    if (old_val <= val) {
        return old_val; // No update needed
    }
    
    do {
        assumed = old;
        // Using explicit comparison instead of min() to avoid ambiguity
        bfloat new_bfloat_val = (old_val <= val) ? old_val : val;
        ushort new_bfloat_ushort = *((thread ushort*)&new_bfloat_val);
        new_val = (assumed & ~mask) | ((uint)new_bfloat_ushort << shift);
        old = atomic_exchange_explicit(base_address, new_val, memory_order_relaxed);
        
        if (assumed == old) {
            return old_val;
        }
        
        old_val_ushort = (old >> shift) & 0xFFFF;
        old_val = *((thread bfloat*)&old_val_ushort);
        if (old_val <= val) {
            return old_val; // Someone else updated with a smaller value
        }
    } while (true);
}
