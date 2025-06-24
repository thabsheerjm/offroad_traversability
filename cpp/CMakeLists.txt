find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_DIR}/include
    ${CMAKE_SOURCE_DIR}/cpp/include  
)

link_directories(
    ${ONNXRUNTIME_DIR}/lib
)

set(SOURCES
    src/main.cpp
    src/preprocessing.cpp
    src/inference.cpp
    src/postprocessing.cpp
)

add_executable(offroad_segmentation ${SOURCES})
target_link_libraries(offroad_segmentation
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
)

add_executable(img2video src/img2video.cpp)
target_link_libraries(img2video ${OpenCV_LIBS})


