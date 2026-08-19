cmake_minimum_required(VERSION 3.16)

project(
    flora_computing_server
    LANGUAGES C CXX
)

# ============================================================
# basic setup
# ============================================================

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(
    CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR}/bin
)

# Default set Release
if(NOT CMAKE_BUILD_TYPE)
    set(
        CMAKE_BUILD_TYPE
        Release
        CACHE STRING
        "Build type"
        FORCE
    )
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Source directory: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Binary directory: ${CMAKE_BINARY_DIR}")

# ============================================================
# OpenMP
# ============================================================

find_package(OpenMP REQUIRED)
# ============================================================
# EMP
# ============================================================

find_path(
    CMAKE_FOLDER
    NAMES
        cmake/emp-tool-config.cmake
    PATHS
        /home/ices/temp_test/local_inst
    NO_DEFAULT_PATH
)

if(NOT CMAKE_FOLDER)
    find_path(
        CMAKE_FOLDER
        NAMES
            cmake/emp-tool-config.cmake
    )
endif()

if(NOT CMAKE_FOLDER)
    message(
        FATAL_ERROR
        "Cannot find EMP installation containing "
        "cmake/emp-tool-config.cmake"
    )
endif()

message(
    STATUS
    "Found EMP CMake folder: ${CMAKE_FOLDER}"
)

# emp-base.cmake will visit:
# ${CMAKE_FOLDER}/cmake/enable_rdseed.cmake
# ${CMAKE_FOLDER}/cmake/enable_crypto_circuit.cmake
# ${CMAKE_FOLDER}/cmake/enable_threading.cmake
include(
    ${CMAKE_FOLDER}/cmake/emp-base.cmake
)

find_package(
    emp-ot
    REQUIRED
)

message(STATUS "EMP-OT include directories: ${EMP-OT_INCLUDE_DIRS}")
message(STATUS "EMP-OT libraries: ${EMP-OT_LIBRARIES}")

# ============================================================\
# shared include path
# program structure:
#
# temp_test/flora-master/
# ├── common/
# ├── secret-share/
# └── computing-server/
# ============================================================

set(
    FLORA_COMMON_INCLUDE_DIRS

    ${CMAKE_CURRENT_BINARY_DIR}

    # computing-server/
    ${CMAKE_CURRENT_SOURCE_DIR}

    # flora-master/
    ${CMAKE_CURRENT_SOURCE_DIR}/..

    # emp local inst
    /home/ices/temp_test/local_inst/include

    ${EMP-OT_INCLUDE_DIRS}
)

# ============================================================
# CPU target shared function
# ============================================================

function(flora_add_cpu_server TARGET_NAME SOURCE_FILE)
    add_executable(
        ${TARGET_NAME}
        ${SOURCE_FILE}
    )

    target_include_directories(
        ${TARGET_NAME}
        PRIVATE
        ${FLORA_COMMON_INCLUDE_DIRS}
    )

    target_compile_options(
        ${TARGET_NAME}
        PRIVATE

        $<$<COMPILE_LANGUAGE:CXX>:
            -O3
            -march=native
            -maes
            -fopenmp
        >
    )

    target_link_libraries(
        ${TARGET_NAME}
        PRIVATE

        ${EMP-OT_LIBRARIES}
        pthread
        OpenMP::OpenMP_CXX
    )

    set_target_properties(
        ${TARGET_NAME}
        PROPERTIES

        RUNTIME_OUTPUT_DIRECTORY
        ${CMAKE_BINARY_DIR}/bin
    )
endfunction()

# ============================================================
# CPU Target
# ============================================================

flora_add_cpu_server(
    server
    server.cpp
)

# ============================================================
# Offline preprocessing: Beaver-triple / GF-correlation generator
# (no EMP dependency; pure C++14)
# ============================================================

add_executable(
    generate_flora_triples
    preprocess/generate_flora_triples.cpp
)

target_include_directories(
    generate_flora_triples
    PRIVATE
    ${FLORA_COMMON_INCLUDE_DIRS}
)

target_compile_options(
    generate_flora_triples
    PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2 -std=c++14>
)

set_target_properties(
    generate_flora_triples
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR}/bin
)

# ============================================================
# GPU Target
# ============================================================

option(
    BUILD_GPU
    "Build server_gpu"
    ON
)

if(BUILD_GPU)
    include(CheckLanguage)

    check_language(CUDA)

    if(NOT CMAKE_CUDA_COMPILER)
        message(
            FATAL_ERROR
            "BUILD_GPU=ON, but CUDA compiler was not found. "
            "Set CUDACXX or pass "
            "-DCMAKE_CUDA_COMPILER=/path/to/nvcc"
        )
    endif()

    enable_language(CUDA)

    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    message(
        STATUS
        "CUDA compiler: ${CMAKE_CUDA_COMPILER}"
    )

    message(
        STATUS
        "CUDA compiler version: ${CMAKE_CUDA_COMPILER_VERSION}"
    )

    # --------------------------------------------------------
    # RTX A6000 = Ampere GA102 = compute capability 8.6
    #
    # CMake 3.16 has incomplete support for CMAKE_CUDA_ARCHITECTURES,
    # so we pass --generate-code directly here.
    #
    # sm_86 can be generated with CUDA 11.1 and later.
    #
    # CUDA 10.1 does not recognize compute_86. As a temporary workaround,
    # we generate compute_75 PTX, which will be JIT-compiled by the NVIDIA
    # driver when running on A6000 GPUs.
    # --------------------------------------------------------
    if(
        CMAKE_CUDA_COMPILER_VERSION
        VERSION_GREATER_EQUAL
        "11.1"
    )
        message(
            STATUS
            "Generating native CUDA code for RTX A6000: "
            "sm_86 + compute_86 PTX"
        )

        set(
            FLORA_CUDA_ARCH_OPTIONS

            "$<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_86,code=sm_86>"
            "$<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_86,code=compute_86>"
        )
    else()
        message(
            WARNING
            "CUDA ${CMAKE_CUDA_COMPILER_VERSION} is too old to "
            "generate native sm_86 code for RTX A6000. "
            "Building compute_75 PTX fallback. "
            "The NVIDIA driver will JIT-compile it for A6000. "
            "For best performance, install CUDA Toolkit 11.8 or 12.2."
        )

        set(
            FLORA_CUDA_ARCH_OPTIONS

            "$<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_75,code=compute_75>"
        )
    endif()

    # --------------------------------------------------------
    # GPU executable file
    # --------------------------------------------------------

    add_executable(
        server_gpu
        server_gpu.cpp
        gpu/gpu_online.cu
        gpu/gpu_transform.cu
        gpu/gpu_shuffle.cu
        gpu/gpu_packed_partition_compare.cu
    )

    target_include_directories(
        server_gpu
        PRIVATE
        ${FLORA_COMMON_INCLUDE_DIRS}
    )

    target_compile_definitions(
        server_gpu
        PRIVATE
    )

    target_compile_options(
        server_gpu
        PRIVATE

        $<$<COMPILE_LANGUAGE:CXX>:
            -march=native
            -maes
            -fopenmp
        >

        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
            -Xcompiler=-fPIC
            -Xcompiler=-fopenmp
            -Xcompiler=-march=native
            -Xcompiler=-maes
        >

        ${FLORA_CUDA_ARCH_OPTIONS}
    )

    target_link_libraries(
        server_gpu
        PRIVATE

        ${EMP-OT_LIBRARIES}

        pthread
        OpenMP::OpenMP_CXX

        # CUDA Runtime
        cudart
    )

    set_target_properties(
        server_gpu
        PROPERTIES

        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF

        RUNTIME_OUTPUT_DIRECTORY
        ${CMAKE_BINARY_DIR}/bin
    )

    message(
        STATUS
        "Configured GPU target: server_gpu"
    )
else()
    message(
        STATUS
        "BUILD_GPU=OFF: skipping server_gpu"
    )
endif()