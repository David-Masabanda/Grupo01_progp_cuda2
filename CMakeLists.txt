cmake_minimum_required(VERSION 3.26)
project(Grupal01_pp2 CUDA)

set(CMAKE_CUDA_STANDARD 17)

enable_language(CXX CUDA)

find_package(fmt)

add_executable(Grupal01_pp2
        Ejercicio2.cu
)

target_include_directories(Grupal01_pp2
        PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(
        Grupal01_pp2 fmt::fmt
        ${CUDART_LIBRARY}
)


set_target_properties(Grupal01_pp2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
