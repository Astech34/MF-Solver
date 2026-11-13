cmake_minimum_required(VERSION 3.16)
project(MF-Solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP)

# Core library
add_library(mfsolver
    src/bands.cpp
    src/constraints.cpp
    src/energy.cpp
    src/hamiltonian.cpp
    src/interaction.cpp
    src/kspace.cpp
    src/observable.cpp
    src/occupations.cpp
    src/output.cpp
    src/scf.cpp
    src/models/kagome.cpp
)
target_include_directories(mfsolver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(mfsolver PUBLIC Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mfsolver PUBLIC OpenMP::OpenMP_CXX)
endif()

# Apps
add_executable(compute_bands apps/compute_bands.cpp)
target_link_libraries(compute_bands PRIVATE mfsolver)

add_executable(single_calculation apps/single_calculation.cpp)
target_link_libraries(single_calculation PRIVATE mfsolver)

add_executable(find_gs apps/find_gs.cpp)
target_link_libraries(find_gs PRIVATE mfsolver)

# Tests (GoogleTest): use system install if present, otherwise download it
option(BUILD_TESTING "Build unit tests" ON)
if(BUILD_TESTING)
    enable_testing()
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    include(GoogleTest)
    file(GLOB TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)
    if(TEST_SOURCES)
        add_executable(mfsolver_tests ${TEST_SOURCES})
        target_link_libraries(mfsolver_tests PRIVATE mfsolver GTest::gtest_main)
        # Miniconda's older libstdc++ can shadow the system one at run time;
        # link the C++ runtime statically so the test binary ignores it.
        target_link_options(mfsolver_tests PRIVATE -static-libstdc++ -static-libgcc)
        gtest_discover_tests(mfsolver_tests DISCOVERY_MODE PRE_TEST)
    endif()
endif()
