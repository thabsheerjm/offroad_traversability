cmake_minimum_required(VERSION 3.10)
project(offroad_traversability)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/onnxruntime")

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

add_subdirectory(cpp)