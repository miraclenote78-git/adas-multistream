cmake_minimum_required(VERSION 3.20)
project(adas_multistream
    VERSION 0.1.0
    DESCRIPTION "Multi-camera ADAS open-loop replay demo on the visionpipe-npu runtime"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- visionpipe-npu (the platform SDK this demo runs on) ---
set(VISIONPIPE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../visionpipe-npu"
    CACHE PATH "Path to the visionpipe-npu source tree")

if(NOT EXISTS "${VISIONPIPE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "visionpipe-npu not found at ${VISIONPIPE_DIR}. "
        "Set -DVISIONPIPE_DIR=/path/to/visionpipe-npu")
endif()

# Only the libraries — no tests/demos/benchmarks from the SDK tree.
set(VPNPU_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
set(VPNPU_BUILD_DEMOS      OFF CACHE BOOL "" FORCE)
set(VPNPU_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
add_subdirectory("${VISIONPIPE_DIR}" visionpipe-npu-build)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -Wshadow -Wconversion -Wsign-conversion
        $<$<CONFIG:Release>:-O3>
    )
endif()

add_executable(multi_stream src/multi_stream.cpp)
target_link_libraries(multi_stream PRIVATE
    visionpipe::vision
    visionpipe::model
)

message(STATUS "")
message(STATUS "adas-multistream configuration:")
message(STATUS "  visionpipe-npu: ${VISIONPIPE_DIR}")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "")
