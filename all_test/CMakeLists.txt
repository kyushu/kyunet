cmake_minimum_required(VERSION 2.9)

project(all_test)

# stb_SRC is already included in kyunet/CMakeList.txt
# so we don't need to include stb header here again.

# kyunet header file
INCLUDE_DIRECTORIES (
	${PROJECT_SOURCE_DIR}/../../inc
	)

file(GLOB SRC *.cpp)
find_package(OpenCV REQUIRED)
    if (OpenCV_FOUND)
        INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

        add_executable(load_imabe_test load_imabe_test.cpp ${stb_SRC})
        target_link_libraries(load_imabe_test ${OpenCV_LIBS})

        add_executable(tensor_test tensor_test.cpp)
        target_link_libraries(tensor_test ${OpenCV_LIBS} libkyunet)

    else()
        message("OpenCV not found")
    endif()