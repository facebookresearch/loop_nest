option(DABUN_BUILD_APPS_FOR_AVX2 "Set to ON to build apps for AVX2 extension" OFF)
option(DABUN_BUILD_APPS_FOR_AVX2_PLUS "Set to ON to build apps for AVX512 extension using AVX512 instructions but only AVX2 (YMM) registers" OFF)
option(DABUN_BUILD_APPS_FOR_AVX512 "Set to ON to build apps for AVX512 extension" OFF)
option(DABUN_BUILD_APPS_FOR_AMX "Set to ON to build apps for AMX extension" OFF)

add_library(dabun
  src/x86/loop_nest.cpp
  src/x86/transposer.cpp
  src/x86/peak_gflops.cpp)

target_include_directories(${PROJECT_NAME}
  PUBLIC ${PROJECT_BINARY_DIR})

target_include_directories(${PROJECT_NAME}
  PUBLIC include)

target_include_directories(${PROJECT_NAME}
  PUBLIC extern/xbyak)
