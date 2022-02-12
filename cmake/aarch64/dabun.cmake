option(DABUN_BUILD_APPS_FOR_NEON "Set to ON to build apps for NEON extension" OFF)
option(DABUN_BUILD_APPS_FOR_NEON_FP16 "Set to ON to build apps for NEON FP16 extension" OFF)

add_library(dabun
  ${DABUN_COMMON_SRC_CPP_FILES})

target_include_directories(${PROJECT_NAME}
  PUBLIC ${PROJECT_BINARY_DIR})

target_include_directories(${PROJECT_NAME}
  PUBLIC include)

target_include_directories(${PROJECT_NAME}
  PUBLIC extern/xbyak_aarch64)

target_compile_options(dabun
  PRIVATE "-DDABUN_COMPILING_LIBDABUN")
