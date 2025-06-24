#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "preprocessing.h"
#include "inference.h"
#include "postprocessing.h"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: ./offroad_segmentation <onnx_model_path> <input_video_path> <output_video_path>\n";
        return -1;
    }
    std::string model_path = argv[1];
    std::string input_video_path = argv[2];
    std::string output_video_path = argv[3];

    cv::VideoCapture cap(input_video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open input video file.\n";
        return -1;
    }

    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));

    cv::VideoWriter writer(output_video_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frame_width, frame_height));
    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open output video for writing.\n";
        return -1;
    }

    InferenceEngine engine(model_path);

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) break;

        auto start_time = std::chrono::high_resolution_clock::now();

        // Preprocess
        std::vector<float> input_tensor;
        cv::Mat resized = preprocess(frame, frame.cols, frame.rows, input_tensor);

        // Inference
        std::vector<float> output_data = engine.run(input_tensor, resized.rows, resized.cols);

        // Postprocess
        cv::Mat mask = postprocess_mask(output_data, resized.rows, resized.cols, 0.5f);
        cv::Mat overlay = overlay_mask(resized, mask,  0.5f);

        // Timing
        auto end_time = std::chrono::high_resolution_clock::now();
        float fps_display = 1000.0f / std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        // Display and write
        cv::putText(overlay, "FPS: " + std::to_string(fps_display), {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 1, {255, 255, 255}, 2);
        writer.write(overlay);
        cv::imshow("Offroad Segmentation", overlay);
        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();
    return 0;
}