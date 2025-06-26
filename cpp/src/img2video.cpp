#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./img2video <image_folder> <output_video_path.mp4>" << std::endl;
        return -1;
    }
    std::string image_folder = argv[1];
    std::string output_video = argv[2];

    std::vector<std::string> image_files;
    for (const auto& entry : fs::directory_iterator(image_folder)) {
        image_files.push_back(entry.path().string());
    }

    std::sort(image_files.begin(), image_files.end());

    if (image_files.empty()) {
        std::cerr << "No images found in folder." << std::endl;
        return -1;
    }

    cv::Mat first = cv::imread(image_files[0]);
    if (first.empty()) {
        std::cerr << "Failed to read first image." << std::endl;
        return -1;
    }

    int width = first.cols;
    int height = first.rows;
    int fps = 10;

    cv::VideoWriter writer(output_video, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));

    if (!writer.isOpened()) {
        std::cerr << "Failed to open video writer. Check codec support." << std::endl;
        return -1;
    }

    for (const auto& file : image_files) {
        cv::Mat img = cv::imread(file);
        if (img.empty()) continue;
        cv::resize(img, img, cv::Size(width, height));
        writer.write(img);
    }
    writer.release();
    std::cout << "Video created at: " << output_video << std::endl;
    return 0;
}
