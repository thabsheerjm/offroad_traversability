#include "inference.h"
#include <stdexcept>
#include <iostream>

InferenceEngine::InferenceEngine(const std::string& model_path): 
    env(ORT_LOGGING_LEVEL_WARNING, "OffroadSegmentation"),
    session_options(),
    session(env, model_path.c_str(), session_options),
    memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {
        for (size_t i = 0; i < session.GetInputCount(); ++i) {
            input_names_str.emplace_back(session.GetInputNameAllocated(i, allocator).get());
        }
        for (size_t i = 0; i < session.GetOutputCount(); ++i) {
            output_names_str.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
        }
        for (const auto& name : input_names_str)
            input_names.push_back(name.c_str());
        for (const auto& name : output_names_str)
            output_names.push_back(name.c_str());
    }

std::vector<float> InferenceEngine::run(const std::vector<float>& input_tensor, int input_height, int input_width) {
    std::array<int64_t, 4> input_dims = {1, 3, input_height, input_width};
    Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(input_tensor.data()), input_tensor.size(), input_dims.data(), 4
    );
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        &input_tensor_ort,
        1,
        output_names.data(),
        1
    );
    float* output_data = output_tensors.front().GetTensorMutableData<float>();
    int output_size = input_height * input_width;
    return std::vector<float>(output_data, output_data + output_size);
}

std::pair<int, int> InferenceEngine::getInputSize() const {
    auto input_tensor_info = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    auto input_dims = input_tensor_info.GetShape();
    return {static_cast<int>(input_dims[2]), static_cast<int>(input_dims[3])};  // height, width
}
