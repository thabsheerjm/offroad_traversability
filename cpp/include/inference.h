#ifndef INFERENCE_H
#define INFERENCE_H

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

class InferenceEngine {
public:
    InferenceEngine(const std::string& model_path);
    std::vector<float> run(const std::vector<float>& input_tensor, int input_height, int input_width);
    std::pair<int, int> getInputSize() const;
private:
    Ort::Env env;
    Ort::Session session;
    Ort::SessionOptions session_options;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info;
    std::vector<std::string> input_names_str;
    std::vector<std::string> output_names_str;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
};

#endif // INFERENCE_H