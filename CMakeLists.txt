cmake_minimum_required(VERSION 3.10)
project(offroad_traversability)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/onnxruntime")

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

add_executable(offroad_segmentation cpp/segment.cpp)

target_link_libraries(offroad_segmentation
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
)

add_executable(img2video cpp/img2video.cpp)
target_link_libraries(img2video ${OpenCV_LIBS})