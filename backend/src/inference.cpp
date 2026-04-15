#include "inference.h"
#include <stdexcept>
#include <iostream>
#include<chrono>

InferenceEngine::InferenceEngine(const std::string& model_path, const std::string& model_id)
    : model_id_(model_id),
      env(ORT_LOGGING_LEVEL_WARNING, "OffroadSegmentation"),
      session_options(),
      memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

    for (size_t i = 0; i < session->GetInputCount(); ++i) {
        input_names_str.emplace_back(session->GetInputNameAllocated(i, allocator).get());
    }
    for (size_t i = 0; i < session->GetOutputCount(); ++i) {
        output_names_str.emplace_back(session->GetOutputNameAllocated(i, allocator).get());
    }
    for (const auto& name : input_names_str)
        input_names.push_back(name.c_str());
    for (const auto& name : output_names_str)
        output_names.push_back(name.c_str());

    std::cout << "ONNX Runtime session created!" << std::endl;
}


InferenceResult InferenceEngine::run(const std::vector<float>& input_tensor, int input_height, int input_width){

    auto t0 = std::chrono::high_resolution_clock::now();

    std::array<int64_t, 4> input_dims = {1, 3, input_height, input_width};
    Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(input_tensor.data()),
        input_tensor.size(), input_dims.data(), 4 );

    auto output_tensors = session->Run(
        Ort::RunOptions{nullptr},
        input_names.data(), &input_tensor_ort, 1,
        output_names.data(), 1    );

    auto t1 = std::chrono::high_resolution_clock::now();

    if (output_tensors.empty()){
        throw std::runtime_error("ONNX inference returned no outputs");
    }
    float* output_data = output_tensors.front().GetTensorMutableData<float>();
    int output_size = input_height * input_width;

    InferenceResult result;
    result.inference_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    result.preprocess_ms = 0.0f;
    result.model_id = model_id_;
    result.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
        t1.time_since_epoch()).count();

    result.traversability_mask = cv::Mat(input_height, input_width, CV_32F, output_data).clone();
    result.confidence = static_cast<float>(cv::mean(result.traversability_mask)[0]);
    result.raw_output = std::vector<float>(output_data, output_data + output_size);
    return result;



}




std::pair<int, int> InferenceEngine::getInputSize() const {
    auto input_tensor_info = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    auto input_dims = input_tensor_info.GetShape();
    return {static_cast<int>(input_dims[2]), static_cast<int>(input_dims[3])};  // height, width
}
