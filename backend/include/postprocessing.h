#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

// Applies sigmoid, threshold, and creates a binary mask
cv::Mat postprocess_mask(const std::vector<float>& output_data, int height, int width, float threshold = 0.5f);
// Overlays the binary mask on the original image
cv::Mat overlay_mask(const cv::Mat& original, const cv::Mat& mask, float alpha = 0.5f);

