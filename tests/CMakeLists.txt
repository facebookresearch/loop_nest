set(DABUN_TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_library(dabun_tests_catch2_main
  catch2_main.cpp)

target_link_libraries(dabun_tests_catch2_main
  PUBLIC Catch2::Catch2)

function(dabun_blanket_test name)
  message(STATUS "dabun_blanket_test ${name}_blanket ${name}.cpp")
  add_executable(${name}_blanket ${name}.cpp)
  target_link_libraries(${name}_blanket
    PUBLIC dabun
    PUBLIC dabun_tests_catch2_main)
endfunction(dabun_blanket_test)

dabun_blanket_test(sentinel)

function(dabun_test name vex float isa)
  message(STATUS "dabun_test ${name}.${vex}.${float} ${name}.cpp")
  add_executable(${name}.${vex}.${float} ${name}.cpp)
  target_link_libraries(${name}.${vex}.${float}
    PUBLIC dabun
    PUBLIC dabun_tests_catch2_main)
  target_compile_options(${name}.${vex}.${float}
    PRIVATE "-DDABUN_ISA=${isa}"
    PRIVATE "-DDABUN_VEX=extension::${vex}"
    PRIVATE "-DDABUN_ARITHMETIC=dabun::${float}")
endfunction(dabun_test)

function(dabun_common_tests vex float isa)
  dabun_test(handpicked_loop_nest_test ${vex} ${float} ${isa})
endfunction(dabun_common_tests)

function(dabun_x86_tests vex float isa)
  dabun_common_tests(${vex} ${float} ${isa})
endfunction(dabun_x86_tests)

function(dabun_arm_tests vex float isa)
  dabun_common_tests(${vex} ${float} ${isa})
endfunction(dabun_arm_tests)

if("${DABUN_HOST_ARCHITECTURE}" STREQUAL "x86_64")
  if (DABUN_BUILD_TESTS_FOR_ALL_ARCH_VEX OR DABUN_BUILD_TESTS_FOR_AVX2)
    dabun_x86_tests(avx2 fp32 avx2)
  endif()
  if (DABUN_BUILD_TESTS_FOR_ALL_ARCH_VEX OR DABUN_BUILD_TESTS_FOR_AVX512)
    dabun_x86_tests(avx512 fp32 avx512)
  endif()
elseif("${DABUN_HOST_ARCHITECTURE}" STREQUAL "aarch64")
  if (DABUN_BUILD_APPS_FOR_ALL_SUPPORTED_VEX OR DABUN_BUILD_APPS_FOR_NEON)
    dabun_arm_tests(neon fp32 aarch64)
  endif()
  if (DABUN_BUILD_APPS_FOR_ALL_SUPPORTED_VEX OR DABUN_BUILD_APPS_FOR_NEON_FP16)
    dabun_arm_tests(neon_fp16 fp16 aarch64)
  endif()
endif()
