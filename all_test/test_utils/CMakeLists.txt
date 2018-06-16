cmake_minimum_required(VERSION 3.1)

project(test_utils)

# Set compile mode to debug
# set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_BUILD_TYPE Debug)

add_library(test_utils test_utils.h test_utils.cpp )
