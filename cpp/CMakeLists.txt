
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
    onnxruntime
)

target_include_directories(offroad_segmentation PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_DIR}/include
    ${CMAKE_SOURCE_DIR}/cpp/include
)

add_executable(img2video src/img2video.cpp)
target_link_libraries(img2video ${OpenCV_LIBS})


install(TARGETS offroad_segmentation RUNTIME DESTINATION bin)
install(TARGETS img2video RUNTIME DESTINATION bin)