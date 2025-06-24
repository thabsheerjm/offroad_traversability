#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat preprocess(const cv::Mat& frame, int target_width, int target_height, std::vector<float>& output_tensor);
