
cmake_minimum_required(VERSION 2.8.12)
project(cudnn_test)
# include(../use_cpp_11.cmake)

# Override the FindCUDA.cmake setting to avoid duplication of host flags if using a toolchain:
option(CUDA_PROPAGATE_HOST_FLAGS "Propage C/CXX_FLAGS and friends to the host compiler via -Xcompile" OFF)
find_package(CUDA 7.5 REQUIRED)
set(CUDA_HOST_COMPILATION_CPP ON)
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30;-std=c++11;-D__STRICT_ANSI__")
add_definitions(-DKYUNET_USE_CUDA)

# include(find_cudnn.txt)

message(STATUS "Looking for cuDNN install...")
# Look for cudnn, we will look in the same place as other CUDA
# libraries and also a few other places as well.
find_path(cudnn_include cudnn.h
    HINTS ${CUDA_INCLUDE_DIRS} ENV CUDNN_INCLUDE_DIR  ENV CUDNN_HOME
    PATHS /usr/local ENV CPATH
    PATH_SUFFIXES include
    )
get_filename_component(cudnn_hint_path "${CUDA_CUBLAS_LIBRARIES}" PATH)
find_library(cudnn cudnn
    HINTS ${cudnn_hint_path} ENV CUDNN_LIBRARY_DIR  ENV CUDNN_HOME 
    PATHS /usr/local /usr/local/cuda ENV LD_LIBRARY_PATH
    PATH_SUFFIXES lib64 lib x64
    )
mark_as_advanced(cudnn cudnn_include)

if (cudnn AND cudnn_include)
    message(STATUS "Found cuDNN: " ${cudnn})

    include_directories(${cudnn_include})
    # cuda_add_library(cudnn_test STATIC ../../cuda/cudnn_dlibapi.cpp ${cudnn} )
    cuda_add_library(cudnn_test STATIC test_cudnn.cpp ${cudnn} )
    # enable_cpp11_for_target(cudnn_test)

else()
    message(STATUS "*** cuDNN V5.0 OR GREATER NOT FOUND.                                                       ***")
    message(STATUS "*** Dlib requires cuDNN V5.0 OR GREATER.  Since cuDNN is not found DLIB WILL NOT USE CUDA. ***")
    message(STATUS "*** If you have cuDNN then set CMAKE_PREFIX_PATH to include cuDNN's folder.                ***")
endif()


# if (cudnn_include AND cudnn)
#    include_directories(${cudnn_include})
#    # cuda_add_library(cudnn_test STATIC ../../cuda/cudnn_dlibapi.cpp ${cudnn} )
#    cuda_add_library(cudnn_test STATIC test_cudnn.cpp ${cudnn} )
#    # enable_cpp11_for_target(cudnn_test)
# endif()
