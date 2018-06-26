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
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
# Set compile mode to debug
set(CMAKE_BUILD_TYPE RelWithDebInfo)


set(USE_CUDA)




# TURN ON/OFF MKTASSERT
add_definitions(-DMKTASSERT)

# Set the output directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#####################################################
# STB image processing library
#####################################################
set(STB ${PROJECT_SOURCE_DIR}/thirdparty/stb)
file(GLOB stb_SRC "${STB}/*.h")
message(STATUS ${stb_SRC})
include_directories(${STB})
# add_library(STB INTERFACE)
# target_sources(STB INTERFACE ${stb_SRC})
# target_include_directories(STB INTERFACE STB)


#####################################################
# CUDA setting: Reference to dlib and need to modify
#####################################################
include(cmake_utils/test_for_cudnn/find_cudnn.txt)
find_package(CUDA)
if(CUDA_FOUND)
    set(CUDA_TEST_CMAKE_FLAGS 
                  "-DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
                  "-DCMAKE_INCLUDE_PATH=${CMAKE_INCLUDE_PATH}"
                  "-DCMAKE_LIBRARY_PATH=${CMAKE_LIBRARY_PATH}")
    
    try_compile(cuda_test_compile_worked 
        ${PROJECT_BINARY_DIR}/cuda_test_build 
        ${PROJECT_SOURCE_DIR}/cmake_utils/test_for_cuda cuda_test
        CMAKE_FLAGS ${CUDA_TEST_CMAKE_FLAGS}
        OUTPUT_VARIABLE try_compile_output_message
    )
   if (NOT cuda_test_compile_worked)
        string(REPLACE "\n" "\n   ***   " try_compile_output_message "${try_compile_output_message}")
        message(STATUS "*****************************************************************************************************************")
        message(STATUS "*** CUDA was found but your compiler failed to compile a simple CUDA program so kyunet isn't going to use CUDA. ")
        message(STATUS "*** The output of the failed CUDA test compile is shown below: ")
        message(STATUS "***   ${try_compile_output_message}")
        message(STATUS "*****************************************************************************************************************")
    else()
        message(STATUS "*******************************")
        message(STATUS " CUDA was found and supported")
        message(STATUS "*******************************")
    endif()

    # if cudnn_test_compile_worked means cudnn is found
    try_compile(cudnn_test_compile_worked 
        ${PROJECT_BINARY_DIR}/cudnn_test_build 
        ${PROJECT_SOURCE_DIR}/cmake_utils/test_for_cudnn cudnn_test
        CMAKE_FLAGS ${CUDA_TEST_CMAKE_FLAGS}
    )
    # if (NOT cudnn_test_compile_worked)
    #     message(STATUS "*** Found cuDNN, but it looks like the wrong version so dlib will not use it.              ***")
    #     message(STATUS "*** Dlib requires cuDNN V5.0 OR GREATER.  Since cuDNN is not found DLIB WILL NOT USE CUDA. ***")
    #     message(STATUS "*** If you have cuDNN then set CMAKE_PREFIX_PATH to include cuDNN's folder.                ***")
    # else()
    #     message(STATUS "*******************************")
    #     message(STATUS " cuDNN was found and supported")
    #     message(STATUS "*******************************")
    # endif()

    
endif(CUDA_FOUND)


#####################################################
# kyunet 
#####################################################
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
