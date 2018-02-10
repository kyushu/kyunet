cmake_minimum_required(VERSION 2.9)

project(all_test)

# Set compile mode to debug
set(CMAKE_BUILD_TYPE RelWithDebInfo)

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

        # load_imabe_test
        add_executable(load_image_test 01_load_image_test.cpp ${stb_SRC})
        target_link_libraries(load_image_test ${OpenCV_LIBS})

        # tensor_test
        add_executable(tensor_test 02_tensor_test.cpp)
        target_link_libraries(tensor_test ${OpenCV_LIBS} libkyunet)

        # test_add_data_flatten
        add_executable(test_add_data_flatten 03_test_add_data_flatten.cpp)
        target_link_libraries(test_add_data_flatten ${OpenCV_LIBS} libkyunet)

        # test_denseLayer
        add_executable(test_denseLayer 05_test_denseLayer.cpp)
        target_link_libraries(test_denseLayer ${OpenCV_LIBS} libkyunet)

        # test_operation
        add_executable(test_operation 06_test_operation.cpp)
        target_link_libraries(test_operation ${OpenCV_LIBS} libkyunet)

        # im2col_test
        add_executable(im2col_test 07_im2col_test.cpp)
        target_link_libraries(im2col_test libkyunet)

        # test_convlayer
        add_executable(test_convlayer 08_test_convlayer.cpp)
        target_link_libraries(test_convlayer ${OpenCV_LIBS} libkyunet)

        # sim_conv_backprop
        add_executable(sim_conv_backprop 09_sim_conv_backprop.cpp)
        target_link_libraries(sim_conv_backprop libkyunet)

        # test_distribute
        add_executable(test_distribute 10_test_distribute.cpp)
        target_link_libraries(test_distribute)

        # test_filler
        add_executable(test_filler 11_test_filler.cpp)
        target_link_libraries(test_filler libkyunet)

    else()
        message("OpenCV not found")
    endif()
