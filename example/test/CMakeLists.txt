cmake_minimum_required(VERSION 2.9)

project(test_prog)

set(STB ${PROJECT_SOURCE_DIR}/../../thirdparty/stb)
MESSAGE(STATUS "${STB}")
file(GLOB stb_SRC "${STB}/*.h")
include_directories(${STB})
link_directories(${STB})


file(GLOB SRC *.cpp)



find_package(OpenCV REQUIRED)
    if (OpenCV_FOUND)
        INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

        add_executable(test_prog ${SRC} ${stb_SRC})
        target_link_libraries(test_prog ${OpenCV_LIBS})

    else()
        message("OpenCV not found, so we won't build the webcam_face_pose_ex example.")
    endif()