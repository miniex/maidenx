cmake_minimum_required(VERSION 3.15)
project(maidenx_be_cuda LANGUAGES CUDA C CXX)

# Set C++17 as the minimum required standard for all C++ code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Feature flag for NN (default OFF)
option(BUILD_NN "Build the neural network library" OFF)

# Optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

#  Add kernels and headers directories to include path
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)

# Modified function to add a CUDA library with improved header handling
function(add_cuda_library LIB_NAME SOURCES)
    # Find .cu source files and their corresponding .cuh files
    set(ALL_FILES "")
    foreach(SOURCE ${SOURCES})
        # Add the source file
        list(APPEND ALL_FILES ${SOURCE})

        # Get the base name without extension
        get_filename_component(SOURCE_NAME ${SOURCE} NAME_WE)
        get_filename_component(SOURCE_DIR ${SOURCE} DIRECTORY)

        # Look for corresponding .cuh files in the source directory
        file(GLOB CUH_FILES
            ${SOURCE_DIR}/${SOURCE_NAME}.cuh
            ${SOURCE_DIR}/*.cuh
        )

        # Look for corresponding .cuh files in the _headers directory
        file(GLOB HEADER_CUH_FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/_headers/${SOURCE_NAME}.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/_headers/*.cuh
        )

        list(APPEND ALL_FILES ${CUH_FILES} ${HEADER_CUH_FILES})
    endforeach()

    # Remove duplicates
    list(REMOVE_DUPLICATES ALL_FILES)

    add_library(${LIB_NAME} STATIC ${ALL_FILES})
    set_target_properties(${LIB_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )
    target_link_libraries(${LIB_NAME}
        PRIVATE
        ${CUDA_LIBRARIES}
        ${CUDA_CUDA_LIBRARY}
        ${CUDA_CUDART_LIBRARY}
    )
endfunction()

# Add "ops" library
file(GLOB OPS_SOURCES
    kernels/ops/*.cu
)
add_cuda_library(ops "${OPS_SOURCES}")
set_target_properties(ops PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

# Add "nn" library
if(BUILD_NN)
    message(STATUS "Building with neural network support")
    file(GLOB_RECURSE NN_SOURCES
        kernels/nn/*.cu
    )
    add_cuda_library(nn "${NN_SOURCES}")
    set_target_properties(nn PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
    )
else()
    message(STATUS "Neural network support disabled")
endif()
