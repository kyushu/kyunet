cmake_minimum_required(VERSION 3.1)

project(all_test)

# Set compile mode to debug
# set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_BUILD_TYPE Debug)

# stb_SRC is already included in kyunet/CMakeList.txt
# so we don't need to include stb header here again.

# kyunet header file
INCLUDE_DIRECTORIES (
	${PROJECT_SOURCE_DIR}/../inc
	)

message(STATUS ${PROJECT_SOURCE_DIR}/../inc/)

# file(GLOB SRC *.cpp)

add_subdirectory(test_utils)
include_directories(test_utils)

macro(add_example name)
   add_executable(${name} ${name}.cpp)
   target_link_libraries(${name} test_utils libkyunet )
endmacro()

macro(add_example_with_stb name)
    add_executable(${name} ${name}.cpp ${stb_SRC})
    target_link_libraries(${name} test_utils libkyunet )
endmacro()

macro(add_example_with_stb_cv name)
    add_executable(${name} ${name}.cpp ${stb_SRC})
    target_link_libraries(${name} test_utils ${OpenCV_LIBS} libkyunet )
endmacro()

find_package(OpenCV REQUIRED)
    if (OpenCV_FOUND)
        INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

        # 01 load_imabe_test
        add_example_with_stb_cv(01_load_image_test)

        # 02 tensor_test
        add_example_with_stb_cv(02_tensor_test)

        # 03 test_add_data_flatten
        add_example_with_stb_cv(03_test_add_data_flatten)

        # # 04 net test
        # add_example(04_net_test)

        # 05 test_denseLayer
        add_example_with_stb_cv(05_test_denseLayer_forward)

        # 06 test_operation
        add_example_with_stb_cv(06_test_operation)

        # 07 im2col_test
        add_example(07_im2col_test)

        # 08 test_convlayer
        add_example_with_stb_cv(08_test_convlayer_forward)

        # 09 sim_conv_backprop
        add_example(09_sim_conv_backprop)

        # 10 test_distribute
        add_example(10_test_distribute)

        # 11 test_filler
        add_example(11_test_filler)

        # 12 test_relu_layer_forward
        add_example(12_test_relu_layer_forward)

        # 13 test_sigmoid_layer_forward
        add_example(13_test_sigmoid_layer_forward)

        # 14 test_pooling_method
        add_example(14_test_pooling_forward)

        # 15 test_softmax_layer
        add_example(15_test_softmax_layer)

        # 16 test_dense_softmax_layer
        add_example(16_test_dense_softmax_layer)

        # 17 test_cross_entropy_with_softmax_layer
        add_example(17_test_cross_entropy_with_softmax_layer)

        # 18 test_cross_entropy_with_softmax_layer
        add_example_with_stb_cv(18_test_pooling_backward)

        # 19_test_convLayer_backward
        add_example(19_test_convLayer_backward)

        # 20_test_conv_pooling_dense_crossLoss
        add_example(20_test_conv_pooling_dense_crossLoss)

        # 21_test_mnist_load_batch
        add_example_with_stb_cv(21_test_mnist_load_batch)

        # 21_test_mnist_load_batch
        add_example_with_stb_cv(22_test_mnist_load_batch_add_batchNorm)

    else()
        message("OpenCV not found")
    endif()
