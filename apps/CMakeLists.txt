set(DABUN_APPS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

function(dabun_extension_specific_app name vex float isa)
  message(STATUS "dabun_extension_specific_app ${name}.${vex}.${float} ${name}.cpp")
  add_executable(${name}.${vex}.${float} ${name}.cpp)
  target_link_libraries(${name}.${vex}.${float}
    PUBLIC dabun
    PUBLIC -ldl
    PUBLIC -lpthread)
  target_compile_options(${name}.${vex}.${float}
    PRIVATE "-DDABUN_ISA=${isa}"
    PRIVATE "-DDABUN_VEX=extension::${vex}"
    PRIVATE "-DDABUN_ARITHMETIC=dabun::${float}")
endfunction(dabun_extension_specific_app)

function(dabun_app name)
  message(STATUS "dabun_app ${name}.cpp ${name}.cpp")
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name}
    PUBLIC dabun
    PUBLIC -ldl
    PUBLIC -lpthread)
endfunction(dabun_app)

function(dabun_x86_apps vex float isa)
  dabun_extension_specific_app(address_packer_search ${vex} ${float} ${isa})
  dabun_extension_specific_app(gflops ${vex} ${float} ${isa})
  dabun_extension_specific_app(loop_nest ${vex} ${float} ${isa})
  dabun_extension_specific_app(loop_nest_bench ${vex} ${float} ${isa})
  dabun_extension_specific_app(loop_nest_stress_test ${vex} ${float} ${isa})
  dabun_extension_specific_app(loop_nest_tests ${vex} ${float} ${isa})
  dabun_extension_specific_app(loop_tree ${vex} ${float} ${isa})
  dabun_extension_specific_app(serialization ${vex} ${float} ${isa})
  dabun_extension_specific_app(transposer ${vex} ${float} ${isa})
  dabun_extension_specific_app(transposer_stress_test ${vex} ${float} ${isa})
endfunction(dabun_x86_apps)

function(dabun_arm_apps vex float isa)
  dabun_extension_specific_app(gflops ${vex} ${float} ${isa})
  dabun_extension_specific_app(loop_nest ${vex} ${float} ${isa})
  dabun_extension_specific_app(loop_nest_bench ${vex} ${float} ${isa})
  dabun_extension_specific_app(loop_nest_stress_test ${vex} ${float} ${isa})
  dabun_extension_specific_app(loop_nest_tests ${vex} ${float} ${isa})
#  dabun_extension_specific_app(loop_tree ${vex} ${float} ${isa})
#  dabun_extension_specific_app(serialization ${vex} ${float} ${isa})
#  dabun_extension_specific_app(transposer ${vex} ${float} ${isa})
#  dabun_extension_specific_app(transposer_stress_test ${vex} ${float} ${isa})
endfunction(dabun_arm_apps)

dabun_app(tensillica_play)
dabun_app(loop_nest_tensillica)

if("${DABUN_HOST_ARCHITECTURE}" STREQUAL "x86_64")
  if (DABUN_BUILD_APPS_FOR_ALL_SUPPORTED_VEX OR DABUN_BUILD_APPS_FOR_AVX2)
    dabun_x86_apps(avx2 fp32_t avx2)
  endif()
  if (DABUN_BUILD_APPS_FOR_ALL_SUPPORTED_VEX OR DABUN_BUILD_APPS_FOR_AVX512)
    dabun_x86_apps(avx512 fp32_t avx512)
    dabun_extension_specific_app(gflops avx512_ymm fp32_t avx2_plus)
  endif()
elseif("${DABUN_HOST_ARCHITECTURE}" STREQUAL "aarch64")
  if (DABUN_BUILD_APPS_FOR_ALL_SUPPORTED_VEX OR DABUN_BUILD_APPS_FOR_NEON)
    dabun_arm_apps(neon fp32_t aarch64)
  endif()
  if (DABUN_BUILD_APPS_FOR_ALL_SUPPORTED_VEX OR DABUN_BUILD_APPS_FOR_NEON_FP16)
    dabun_arm_apps(neon_fp16 fp16_t aarch64)
  endif()
elseif("${DABUN_HOST_ARCHITECTURE}" STREQUAL "arm64")
  if (DABUN_BUILD_APPS_FOR_ALL_SUPPORTED_VEX OR DABUN_BUILD_APPS_FOR_NEON)
    dabun_arm_apps(neon fp32_t aarch64)
  endif()
  if (DABUN_BUILD_APPS_FOR_ALL_SUPPORTED_VEX OR DABUN_BUILD_APPS_FOR_NEON_FP16)
    dabun_arm_apps(neon_fp16 fp16_t aarch64)
  endif()
else()
endif()
