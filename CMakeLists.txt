cmake_minimum_required(VERSION 3.20)
project(MiniMind VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 统一输出目录到项目根目录下的 bin 文件夹
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

# 查找并链接 OpenMP（多核并行加速）
find_package(OpenMP)

add_executable(minimind_cpp src/minimind.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(minimind_cpp PRIVATE OpenMP::OpenMP_CXX)
endif()

# 开启性能优化标记与架构指令集（如 AVX2/FMA 等）
if(MSVC)
    target_compile_options(minimind_cpp PRIVATE /O2 /W4)
else()
    target_compile_options(minimind_cpp PRIVATE -O3 -Wall -Wextra -march=native)
endif()
