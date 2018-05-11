cmake_minimum_required(VERSION 3.1)
project(kyunet)

# Check compiler
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    message(STATUS "Use C++11")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# Debug setting
SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
# Set compile mode to debug
set(CMAKE_BUILD_TYPE RelWithDebInfo)




# TURN ON/OFF MKTASSERT
add_definitions(-DMKTASSERT)

# Set the output directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# STB
set(STB ${PROJECT_SOURCE_DIR}/thirdparty/stb)
file(GLOB stb_SRC "${STB}/*.h")
message(STATUS ${stb_SRC})
include_directories(${STB})
# add_library(STB INTERFACE)
# target_sources(STB INTERFACE ${stb_SRC})
# target_include_directories(STB INTERFACE STB)


include_directories(
	${PROJECT_SOURCE_DIR}/inc
	${PROJECT_SOURCE_DIR}/utils)


# Add all source file
file(GLOB_RECURSE LIB_KYUNET_SRC src/*.cpp)
# add_library(libkyunet STATIC ${LIB_KYUNET_SRC} ${stb_SRC})
add_library(libkyunet STATIC ${LIB_KYUNET_SRC})
target_link_libraries(libkyunet)


#Build examples
set(BUILD_EXAMPLES 1)
set(BUILD_TEST 1)
# if(BUILD_EXAMPLES)
# 	add_subdirectory(example/)
# endif(BUILD_EXAMPLES)

if(BUILD_TEST)
	add_subdirectory(all_test)
endif(BUILD_TEST)
