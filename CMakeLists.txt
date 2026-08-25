cmake_minimum_required(VERSION 3.20)
project(MiniMind-Next VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 统一输出目录到项目根目录下的 bin 文件夹
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

# 查找并链接 OpenMP（多核并行加速）
find_package(OpenMP)

# 源文件列表
set(SOURCES
    src/main.cpp
    src/ops/math_ops.cpp
    src/tokenizer/tokenizer.cpp
    src/tokenizer/mmap_tokenizer.cpp
    src/sampler/sampler.cpp
    src/model/model.cpp
    src/model/embedding.cpp
)

# 添加头文件包含路径
include_directories(src)

add_executable(minimind_cpp ${SOURCES})

# 独立测试目标 (Quantized Embedding & MMap Tokenizer)
add_executable(test_modules
    src/test_modules.cpp
    src/model/embedding.cpp
    src/tokenizer/mmap_tokenizer.cpp
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(minimind_cpp PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(test_modules PRIVATE OpenMP::OpenMP_CXX)
endif()

# 开启性能优化标记与架构指令集（如 AVX2/FMA 等）
if(MSVC)
    target_compile_options(minimind_cpp PRIVATE /O2 /W4)
    target_compile_options(test_modules PRIVATE /O2 /W4)
else()
    target_compile_options(minimind_cpp PRIVATE -O3 -Wall -Wextra -march=native)
    target_compile_options(test_modules PRIVATE -O3 -Wall -Wextra -march=native)
endif()
