#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>

struct InferenceResult {
    cv::Mat traversability_mask;
    float confidence;
    float inference_ms;
    float preprocess_ms;
    int64_t timestamp_us; // micro-seconds
    std::string model_id;
    std::vector<float> raw_output;
};


class InferenceEngine {
public:
    InferenceEngine(const std::string& model_path, const std::string& model_id = "default");
    InferenceResult run(const std::vector<float>& input_tensor, int input_height, int input_width);
    std::pair<int, int> getInputSize() const;

private:
    std::string model_id_;
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::AllocatorWithDefaultOptions allocator;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memory_info;

    std::vector<std::string> input_names_str;
    std::vector<std::string> output_names_str;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
};
