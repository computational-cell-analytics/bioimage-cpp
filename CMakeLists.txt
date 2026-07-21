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
    src/bindings/affinities.cxx
    src/bindings/blocking.cxx
    src/bindings/distance.cxx
    src/bindings/module.cxx
    src/bindings/filters.cxx
    src/bindings/flow.cxx
    src/bindings/graph.cxx
    src/bindings/ground_truth.cxx
    src/bindings/label_multiset.cxx
    src/bindings/mesh.cxx
    src/bindings/segmentation.cxx
    src/bindings/skeleton.cxx
    src/bindings/skeleton_distributed.cxx
    src/bindings/transformation.cxx
    src/bindings/util.cxx
    src/bindings/utils.cxx
    src/cpp/segmentation/mutex_watershed.cxx
    src/cpp/segmentation/semantic_mutex_watershed.cxx
    src/cpp/take_dict.cxx
)

target_include_directories(_core PRIVATE include)
target_compile_features(_core PRIVATE cxx_std_20)
target_compile_options(_core PRIVATE
    $<$<NOT:$<CONFIG:Debug>>:-O3>
)

option(
    BIOIMAGE_FLOW_FMA_DISPATCH
    "Build an x86 FMA-specialized flow tracer with runtime CPU dispatch"
    ON
)
if(BIOIMAGE_FLOW_FMA_DISPATCH)
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _bioimage_system_processor)
    if(_bioimage_system_processor MATCHES "^(x86_64|amd64|i[3-6]86)$")
        if(MSVC)
            target_sources(_core PRIVATE src/cpp/flow/flow_density_fma.cxx)
            set_property(
                SOURCE src/cpp/flow/flow_density_fma.cxx
                APPEND PROPERTY COMPILE_OPTIONS /arch:AVX2
            )
            target_compile_definitions(
                _core PRIVATE
                BIOIMAGE_FLOW_FMA_DISPATCH=1
                BIOIMAGE_FLOW_FMA_REQUIRES_AVX2=1
            )
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_sources(_core PRIVATE src/cpp/flow/flow_density_fma.cxx)
            set_property(
                SOURCE src/cpp/flow/flow_density_fma.cxx
                APPEND PROPERTY COMPILE_OPTIONS -mavx -mfma
            )
            target_compile_definitions(_core PRIVATE BIOIMAGE_FLOW_FMA_DISPATCH=1)
        endif()
    endif()
endif()

option(BIOIMAGE_PROFILE "Enable per-phase profiling instrumentation (development only)" OFF)
if(BIOIMAGE_PROFILE)
    target_compile_definitions(_core PRIVATE BIOIMAGE_PROFILE)
endif()

install(TARGETS _core LIBRARY DESTINATION bioimage_cpp)
