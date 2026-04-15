#pragma once
#include <string>
#include <opencv2/opencv.hpp>

class FrameReader {
public:
    explicit FrameReader(const std::string& source);
    ~FrameReader();

    bool next(cv::Mat& frame);
    int width() const;
    int height() const;
    int fps() const;
    bool isOpen() const;

private:
    cv::VideoCapture cap_;

};
