#include "preprocessing.h"

cv::Mat preprocess(const cv::Mat& frame, int target_width, int target_height, std::vector<float>& output_tensor) {
    cv::Mat resized, rgb;
    cv::resize(frame, resized, cv::Size(target_width, target_height));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    //output tensor in CHW - flat
    output_tensor.resize(3 * target_height * target_width);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < target_height; ++y) {
            for (int x = 0; x < target_width; ++x) {
                output_tensor[c * target_height * target_width + y * target_width + x] =
                    rgb.at<cv::Vec3f>(y, x)[c];
            }
        }
    }

    return resized;
}