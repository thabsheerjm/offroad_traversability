#include "postprocessing.h"
#include "config.h"

cv::Mat postprocess_mask(const std::vector<float>& output_data, int height, int width, float threshold) {

    // rebuilt logits map
    cv::Mat logits(height, width, CV_32F, const_cast<float*>(output_data.data()));
    // apply gblur on raw logits
    cv::Mat blurred;
    cv::GaussianBlur(logits, blurred, cv::Size(5,5), 1.0);

    cv::Mat mask(height, width, CV_8UC1);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float val = 1.0f / (1.0f + std::exp(-blurred.at<float>(y, x))); // Sigmoid
            mask.at<uchar>(y, x) = (val > threshold) ? 255 : 0;
        }
    }
    // remove outlier patches - connected component analysis(CCA)
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(mask, labels, stats, centroids);

    cv::Mat clean_mask = cv::Mat::zeros(height, width, CV_8UC1);
    for (int i =1; i < num_labels; ++i ){
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area >= kMinArea){
            clean_mask.setTo(255, labels == i);
        }
    }
    // don't mask upper area on the image
    int cutoff_row = static_cast<int>(height * kUpperMaskFraction);
    clean_mask(cv::Rect(0,0, width, cutoff_row)).setTo(0);
    return clean_mask;
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
