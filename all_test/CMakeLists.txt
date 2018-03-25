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

file(GLOB SRC *.cpp)
find_package(OpenCV REQUIRED)
    if (OpenCV_FOUND)
        INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

        # 01 load_imabe_test
        add_executable(load_image_test 01_load_image_test.cpp ${stb_SRC})
        target_link_libraries(load_image_test ${OpenCV_LIBS})

        # 02 tensor_test
        add_executable(tensor_test 02_tensor_test.cpp)
        target_link_libraries(tensor_test ${OpenCV_LIBS} libkyunet)

        # 03 test_add_data_flatten
        add_executable(test_add_data_flatten 03_test_add_data_flatten.cpp)
        target_link_libraries(test_add_data_flatten ${OpenCV_LIBS} libkyunet)

        # 05 test_denseLayer
        add_executable(test_denseLayer_forward 05_test_denseLayer_forward.cpp)
        target_link_libraries(test_denseLayer_forward ${OpenCV_LIBS} libkyunet)

        # 06 test_operation
        add_executable(test_operation 06_test_operation.cpp)
        target_link_libraries(test_operation ${OpenCV_LIBS} libkyunet)

        # 07 im2col_test
        add_executable(im2col_test 07_im2col_test.cpp)
        target_link_libraries(im2col_test libkyunet)

        # 08 test_convlayer
        add_executable(test_convlayer_forward 08_test_convlayer_forward.cpp)
        target_link_libraries(test_convlayer_forward ${OpenCV_LIBS} libkyunet)

        # 09 sim_conv_backprop
        add_executable(sim_conv_backprop 09_sim_conv_backprop.cpp)
        target_link_libraries(sim_conv_backprop libkyunet)

        # 10 test_distribute
        add_executable(test_distribute 10_test_distribute.cpp)
        target_link_libraries(test_distribute)

        # 11 test_filler
        add_executable(test_filler 11_test_filler.cpp)
        target_link_libraries(test_filler libkyunet)

        # 12 test_relu_layer_forward
        add_executable(test_relu_layer_forward 12_test_relu_layer_forward.cpp)
        target_link_libraries(test_relu_layer_forward libkyunet)

        # 13 test_sigmoid_layer_forward
        add_executable(test_sigmoid_layer_forward 13_test_sigmoid_layer_forward.cpp)
        target_link_libraries(test_sigmoid_layer_forward libkyunet)

        # 14 test_pooling_method
        add_executable(test_pooling_forward 14_test_pooling_forward.cpp)
        target_link_libraries(test_pooling_forward libkyunet)

        # 15 test_softmax_layer
        add_executable(test_softmax_layer 15_test_softmax_layer.cpp)
        target_link_libraries(test_softmax_layer libkyunet)

        # 16 test_dense_softmax_layer
        add_executable(test_dense_softmax_layer 16_test_dense_softmax_layer.cpp)
        target_link_libraries(test_dense_softmax_layer libkyunet)

        # 17 test_cross_entropy_with_softmax_layer
        add_executable(test_cross_entropy_with_softmax_layer 17_test_cross_entropy_with_softmax_layer.cpp)
        target_link_libraries(test_cross_entropy_with_softmax_layer libkyunet)

        # 18 test_cross_entropy_with_softmax_layer
        add_executable(test_pooling_backward 18_test_pooling_backward.cpp)
        target_link_libraries(test_pooling_backward libkyunet)

        # 19_test_convLayer_backward
        add_executable(test_convLayer_backward 19_test_convLayer_backward.cpp)
        target_link_libraries(test_convLayer_backward libkyunet)

        # 20_test_conv_pooling_dense_crossLoss
        add_executable(test_conv_pooling_dense_crossLoss 20_test_conv_pooling_dense_crossLoss.cpp)
        target_link_libraries(test_conv_pooling_dense_crossLoss libkyunet)

    else()
        message("OpenCV not found")
    endif()
