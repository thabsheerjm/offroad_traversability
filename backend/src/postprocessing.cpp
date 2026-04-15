#include "postprocessing.h"

cv::Mat postprocess_mask(const std::vector<float>& output_data, int height, int width, float threshold) {
    cv::Mat mask(height, width, CV_8UC1);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float val = 1.0f / (1.0f + std::exp(-output_data[y * width + x])); // Sigmoid
            mask.at<uchar>(y, x) = (val > threshold) ? 255 : 0;
        }
    }
    return mask;
}

cv::Mat overlay_mask(const cv::Mat& original, const cv::Mat& mask, float alpha) {
    cv::Mat color_mask(original.size(), original.type(), cv::Scalar(0, 0, 0));
    for (int y = 0; y < mask.rows; ++y) {
        for (int x = 0; x < mask.cols; ++x) {
            if (mask.at<uchar>(y, x) > 0)
                color_mask.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0); //traversable
            else
                color_mask.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 0, 0); 
        }
    }

    cv::Mat blended;
    cv::addWeighted(original, 1 - alpha, color_mask, alpha, 0, blended);
    return blended; 
}
