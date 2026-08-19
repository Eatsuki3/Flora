cmake_minimum_required(VERSION 3.16)

project(
    flora_micro_eval
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(
    CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR}/bin
)

if(NOT CMAKE_BUILD_TYPE)
    set(
        CMAKE_BUILD_TYPE
        Release
        CACHE STRING
        "Build type"
        FORCE
    )
endif()

# ----------------------------------------------------------------------------
# EMP
# ----------------------------------------------------------------------------

find_path(
    CMAKE_FOLDER
    NAMES cmake/emp-tool-config.cmake
)

if(NOT CMAKE_FOLDER)
    message(
        FATAL_ERROR
        "Cannot find EMP installation containing "
        "cmake/emp-tool-config.cmake"
    )
endif()

include(${CMAKE_FOLDER}/cmake/emp-base.cmake)

find_package(emp-ot REQUIRED)

# ----------------------------------------------------------------------------
# shared include paths
#   micro-eval/
#   flora-master/   (common/, secret-share/, computing-server/)
# ----------------------------------------------------------------------------

set(
    FLORA_MICRO_INCLUDE_DIRS
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${EMP-OT_INCLUDE_DIRS}
)

function(flora_add_micro TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})

    target_include_directories(
        ${TARGET_NAME}
        PRIVATE
        ${FLORA_MICRO_INCLUDE_DIRS}
    )

    target_compile_options(
        ${TARGET_NAME}
        PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -maes>
    )

    target_link_libraries(
        ${TARGET_NAME}
        PRIVATE
        ${EMP-OT_LIBRARIES}
        pthread
    )

    set_target_properties(
        ${TARGET_NAME}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
        ${CMAKE_BINARY_DIR}/bin
    )
endfunction()

flora_add_micro(filter   filter.cpp)
flora_add_micro(neighbor neighbor.cpp)
