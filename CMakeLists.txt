cmake_minimum_required(VERSION 3.16)
project(MF-Solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Core library
add_library(mfsolver
    src/bands.cpp
    src/constraints.cpp
    src/energy.cpp
    src/hamiltonian.cpp
    src/interaction.cpp
    src/observable.cpp
    src/models/kagome.cpp
)
target_include_directories(mfsolver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(mfsolver PUBLIC Eigen3::Eigen)

# Apps
add_executable(compute_bands apps/compute_bands.cpp)
target_link_libraries(compute_bands PRIVATE mfsolver)
