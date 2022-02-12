cmake_minimum_required(VERSION 3.15.0)

add_subdirectory(extern/Catch2)

# SET (CMAKE_CXX_COMPILER             "/usr/bin/clang++")

project(dabun
  LANGUAGES CXX
  VERSION 0.0.0)

configure_file(dabun_config.hpp.in config/dabun_config.hpp)

set(CMAKE_CXX_STANDARD 17)

option(DABUN_DEBUG "Set to ON to build debug version" OFF)
option(DABUN_DEBUG_WERROR "Set to ON to enable all warnings in debug mode" ON)
option(DABUN_BUILD_APPS_FOR_ALL_SUPPORTED_VEX "Set to ON to build apps for all supported extensions" ON)
option(DABUN_BUILD_TESTS_FOR_ALL_ARCH_VEX "Set to ON to build tests for all supported extensions" ON)

if (DABUN_DEBUG)
  message(STATUS "Will compile libdabun in debug mode.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
  if (DABUN_DEBUG_WERROR)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-sign-compare")
  endif()
else()
  message(STATUS "Will compile libdabun in release mode.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -O3 -Wall -Wextra -Werror -Wno-sign-compare")
endif()

##
## Find Boost
##
set(Boost_USE_STATIC_LIBS        ON)  # only find static libs
set(Boost_USE_DEBUG_LIBS        ${DABUN_DEBUG})  # ignore debug libs and
set(Boost_USE_RELEASE_LIBS       ON)  # only find release libs
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)
find_package(Boost 1.66.0)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "Can't find boost libraries")
endif()

##
## Detect host architecture
##
set(DABUN_HOST_ARCHITECTURE_SUPPORTED x86_64 aarch64 arm64)

execute_process(COMMAND uname -m
  COMMAND tr -d '\n'
  OUTPUT_VARIABLE DABUN_HOST_ARCHITECTURE)

message(STATUS "Host architecture detected: ${DABUN_HOST_ARCHITECTURE}")

set(DABUN_COMMON_SRC_CPP_FILES
  src/loop_nest.cpp
  src/transposer.cpp
  src/peak_gflops.cpp)

if("${DABUN_HOST_ARCHITECTURE}" STREQUAL "x86_64")
  include(cmake/x86_64/dabun.cmake)
elseif("${DABUN_HOST_ARCHITECTURE}" STREQUAL "aarch64")
  include(cmake/aarch64/dabun.cmake)
elseif("${DABUN_HOST_ARCHITECTURE}" STREQUAL "arm64")
  include(cmake/aarch64/dabun.cmake)
else()
  message(FATAL_ERROR "Host architecture ${DABUN_HOST_ARCHITECTURE} not supported.")
endif()

add_subdirectory(apps)
add_subdirectory(tests)
