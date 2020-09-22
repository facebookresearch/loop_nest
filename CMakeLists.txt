cmake_minimum_required(VERSION 3.7)

set(CMAKE_CXX_STANDARD 17)

set(ISA "avx512" CACHE STRING "ISA to build for")

if ( $ENV{DABUN_ISA} NOT STREQUAL "" )
  set(ISA $ENV{DABUN_ISA})
endif()

message("Building for ${ISA}")

set(CMAKE_CXX_FLAGS "-march=native -DDABUN_ISA=${ISA}")
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/)
set(XBYAK_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/xbyak)

if ( "${ISA}" STREQUAL "aarch64" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLOOP_NEST_ARM=1 -DARM_LOOP_NEST=1")
  set(XBYAK_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/xbyak_aarch64)
endif()

add_executable(mflop mflop.cpp)
target_include_directories(mflop PUBLIC ${INCLUDE_DIR} ${XBYAK_INCLUDE_DIR})

