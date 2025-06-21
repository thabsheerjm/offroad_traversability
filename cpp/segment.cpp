#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <array>

cv::Mat preprocess(const cv::Mat& frame, int width, int height, std::vector<float>& output_tensor) {
    cv::Mat resized, rgb;
    cv::resize(frame, resized, cv::Size(width, height));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255);


    output_tensor.resize(3 * width * height);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                output_tensor[c * height * width + y * width + x] = rgb.at<cv::Vec3f>(y, x)[c];
            }
        }
    }
    return resized;
}

cv::Mat overlay_mask(const cv::Mat& original, const cv::Mat& mask, float alpha = 0.5) {
    cv::Mat color_mask(original.size(), original.type(), cv::Scalar(0, 0, 0));
    for (int y = 0; y < mask.rows; ++y) {
        for (int x = 0; x < mask.cols; ++x) {
            if (mask.at<uchar>(y, x) > 0) color_mask.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0); //traversable
            else color_mask.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 0, 0);
        }
    }
    cv::Mat blended;
    cv::addWeighted(original, 1 - alpha, color_mask, alpha, 0, blended);
    return blended;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: ./offroad_segmentation <onnx_model_path> <input_video_path> <output_video_path>" << std::endl;
        return -1;
    }
    std::string model_path = argv[1];
    std::string input_video = argv[2];
    std::string output_video = argv[3];

    cv::VideoCapture cap(input_video);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open input video file." << std::endl;
        return -1;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));

    cv::VideoWriter writer(output_video, cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open output video for writing." << std::endl;
        return -1;
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Segmentation");
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> input_names_str, output_names_str;
    for (size_t i = 0; i < session.GetInputCount(); ++i)
        input_names_str.push_back(session.GetInputNameAllocated(i, allocator).get());
    for (size_t i = 0; i < session.GetOutputCount(); ++i)
        output_names_str.push_back(session.GetOutputNameAllocated(i, allocator).get());

    std::vector<const char*> input_names, output_names;
    for (const auto& name : input_names_str) input_names.push_back(name.c_str());
    for (const auto& name : output_names_str) output_names.push_back(name.c_str());

    std::vector<float> input_tensor;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "End of video or failed to read frame." << std::endl;
            break;
        }

        int64_t input_height = frame.rows;
        int64_t input_width = frame.cols;

        auto start_time = std::chrono::high_resolution_clock::now();
        cv::Mat resized = preprocess(frame, input_width, input_height, input_tensor);
        std::array<int64_t, 4> input_dims = {1, 3, input_height, input_width};

        Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor.data(), input_tensor.size(), input_dims.data(), 4);

        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor_ort, 1, output_names.data(), 1);
        float* output_data = output_tensors.front().GetTensorMutableData<float>();

        cv::Mat mask(input_height, input_width, CV_8UC1);
        for (int y = 0; y < input_height; ++y) {
            for (int x = 0; x < input_width; ++x) {
                float val = output_data[y * input_width + x];
                mask.at<uchar>(y, x) = (val > 0.5f) ? 255 : 0;
            }
        }

        cv::resize(mask, mask, frame.size(), 0, 0, cv::INTER_NEAREST);
        cv::Mat overlay = overlay_mask(frame, mask);

        auto end_time = std::chrono::high_resolution_clock::now();
        float fps_infer = 1000.0f / std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        cv::putText(overlay, "FPS: " + std::to_string(fps_infer), {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 1, {255, 255, 255}, 2);

        cv::resize(overlay, overlay, cv::Size(width, height));
        writer.write(overlay);
        cv::imshow("Segmentation", overlay);
        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();
    return 0;
}