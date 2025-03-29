# Directory settings
KERNEL_DIR = kernels
OUTPUT_DIR = build
INCLUDE_DIR = $(KERNEL_DIR)

# Output files
OPS_LIB = $(OUTPUT_DIR)/ops.metallib
NN_LIB = $(OUTPUT_DIR)/nn.metallib

# Intermediate files
OPS_DIR = $(OUTPUT_DIR)/ops
NN_DIR = $(OUTPUT_DIR)/nn

# Utils file
ATOMICS = $(KERNEL_DIR)/atomics.metal
ATOMICS_AIR = $(OUTPUT_DIR)/atomics.air
UTILS = $(KERNEL_DIR)/metal_utils.metal
UTILS_AIR = $(OUTPUT_DIR)/metal_utils.air

# Source files
OPS_SOURCES = $(KERNEL_DIR)/ops/binary.metal \
              $(KERNEL_DIR)/ops/matmul.metal \
              $(KERNEL_DIR)/ops/padding.metal \
              $(KERNEL_DIR)/ops/reduction.metal \
              $(KERNEL_DIR)/ops/unary.metal

NN_SOURCES = $(KERNEL_DIR)/nn/activation/softmax.metal \
             $(KERNEL_DIR)/nn/conv.metal

# Derived .air file paths
OPS_AIR_FILES = $(patsubst $(KERNEL_DIR)/ops/%.metal,$(OPS_DIR)/%.air,$(OPS_SOURCES))
NN_AIR_FILES = $(patsubst $(KERNEL_DIR)/nn/%.metal,$(NN_DIR)/%.air,$(NN_SOURCES))
NN_ACTIVATION_AIR_FILES = $(patsubst $(KERNEL_DIR)/nn/activation/%.metal,$(NN_DIR)/activation/%.air,$(wildcard $(KERNEL_DIR)/nn/activation/*.metal))

# Compiler settings
METAL = xcrun -sdk macosx metal
METALLIB = xcrun -sdk macosx metallib
METAL_FLAGS = -I $(INCLUDE_DIR)

# Default target
all: prepare $(OPS_LIB) $(NN_LIB)

# Create build directories
prepare:
	@mkdir -p $(OUTPUT_DIR)
	@mkdir -p $(OPS_DIR)
	@mkdir -p $(NN_DIR)
	@mkdir -p $(NN_DIR)/activation

# Compile compatibility
$(ATOMICS_AIR): $(ATOMICS) $(HEADERS)
	$(METAL) $(METAL_FLAGS) -c $< -o $@

# Compile utils
$(UTILS_AIR): $(UTILS) $(HEADERS)
	$(METAL) $(METAL_FLAGS) -c $< -o $@

# Compile OPS metal files individually
$(OPS_DIR)/%.air: $(KERNEL_DIR)/ops/%.metal $(HEADERS) $(ATOMICS_AIR) $(UTILS_AIR)
	$(METAL) $(METAL_FLAGS) -c $< -o $@

# Compile NN metal files individually
$(NN_DIR)/%.air: $(KERNEL_DIR)/nn/%.metal $(HEADERS) $(ATOMICS_AIR) $(UTILS_AIR)
	$(METAL) $(METAL_FLAGS) -c $< -o $@

$(NN_DIR)/activation/%.air: $(KERNEL_DIR)/nn/activation/%.metal $(HEADERS) $(ATOMICS_AIR) $(UTILS_AIR)
	$(METAL) $(METAL_FLAGS) -c $< -o $@

# Create OPS metallib from all .air files
$(OPS_LIB): $(ATOMICS_AIR) $(UTILS_AIR) $(OPS_AIR_FILES)
	$(METALLIB) $^ -o $@

# Create NN metallib from all .air files
$(NN_LIB): $(ATOMICS_AIR) $(UTILS_AIR) $(NN_AIR_FILES) $(NN_ACTIVATION_AIR_FILES)
	$(METALLIB) $^ -o $@

# Cleanup
clean:
	rm -rf $(OUTPUT_DIR)

# Debugging targets
debug-paths:
	@echo "OPS sources: $(OPS_SOURCES)"
	@echo "OPS AIR files: $(OPS_AIR_FILES)"
	@echo "NN sources: $(NN_SOURCES)"
	@echo "NN AIR files: $(NN_AIR_FILES)"
	@echo "NN activation AIR files: $(NN_ACTIVATION_AIR_FILES)"

# Dependency declarations
.PHONY: all prepare clean debug-paths
