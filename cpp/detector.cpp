#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <array>

std::vector<std::string> loadLabels(const std::string& filename) {
    std::vector<std::string> labels;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        labels.push_back(line);
    }
    return labels;
}

// TODO: Convert to segmentation pipeline for inference

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: ./obstacle_detector <onnx_model_path> <video_path> <labels_txt_path>" << std::endl;
        return -1;
    }
    std::string model_path = argv[1];
    std::string video_path = argv[2];
    std::string labels_path = argv[3];
    std::vector<std::string> labels = loadLabels(labels_path);

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video file." << std::endl;
        return -1;
    }

    // Set up ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ObstacleDetector");
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> input_names_str = session.GetInputNames();
    std::vector<std::string> output_names_str = session.GetOutputNames();

    std::vector<const char*> input_names, output_names;
    for (const auto& name : input_names_str) input_names.push_back(name.c_str());
    for (const auto& name : output_names_str) output_names.push_back(name.c_str());

    auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    size_t input_height = input_shape[2];
    size_t input_width = input_shape[3];

    cv::Mat frame;
    while (cap.read(frame)) {
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(input_width, input_height));
        resized.convertTo(resized, CV_32F, 1.0 / 255);

        std::vector<float> input_tensor_values(resized.total() * resized.channels());
        std::memcpy(input_tensor_values.data(), resized.data, input_tensor_values.size() * sizeof(float));

        std::array<int64_t, 4> input_dims = {1, 3, (int64_t)input_height, (int64_t)input_width};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator.GetInfo(), input_tensor_values.data(), input_tensor_values.size(), input_dims.data(), 4);
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);
        float* raw_output = output_tensors.front().GetTensorMutableData<float>();

        int predicted_class = std::max_element(raw_output, raw_output + 1000) - raw_output; 
        std::string label = (predicted_class < labels.size()) ? labels[predicted_class] : "Unknown";

        cv::putText(frame, "Class: " + label, {30, 30}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 255, 0}, 2);
        cv::imshow("Obstacle Detection", frame);

        if (cv::waitKey(30) >= 0) break;
    }

    return 0;
}
