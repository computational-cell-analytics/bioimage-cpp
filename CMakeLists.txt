cmake_minimum_required(VERSION 3.21)
project(bioimage_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core
    NB_STATIC
    src/bindings/module.cxx
    src/bindings/segmentation.cxx
    src/bindings/utils.cxx
    src/cpp/segmentation/mutex_watershed.cxx
    src/cpp/take_dict.cxx
)

target_include_directories(_core PRIVATE include)
target_compile_features(_core PRIVATE cxx_std_20)

install(TARGETS _core LIBRARY DESTINATION bioimage_cpp)
