#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "config.h"
#include "frame_reader.h"
#include "preprocessing.h"
#include "inference.h"
#include "postprocessing.h"
#include "zmq_pub.h"


int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: ./offroad_segmentation <onnx_model_path> <input_video_path> <output_video_path>\n"
                  << "bridge/bridge.py launches this.";
        return 1;
    }

    std::string model_path = argv[1];
    std::string input_video_path = argv[2];
    std::string output_video_path = argv[3];

    try {
        FrameReader reader(input_video_path);
        InferenceEngine engine(model_path);
        const char* zmq_port_env = std::getenv("OFFROAD_ZMQ_PORT");
        std::string zmq_endpoint = std::string("tcp://*:") + (zmq_port_env ? zmq_port_env: "5555");
        ZMQPub publisher(zmq_endpoint);

        cv::VideoWriter writer(
            output_video_path, cv::VideoWriter::fourcc('m','p','4','v'),
            reader.fps(), cv::Size(inference_width, inference_height)
        );
        if (!writer.isOpened()){
            throw std::runtime_error("Could not open output file for writing: " + output_video_path);
        }

        cv::Mat frame;
        while (reader.next(frame)){
            auto t0 = std::chrono::high_resolution_clock::now();

            std::vector<float> input_tensor;
            cv::Mat resized = preprocess(frame, inference_width, inference_height, input_tensor);

            auto t1 = std::chrono::high_resolution_clock::now();
            InferenceResult result = engine.run(input_tensor, resized.rows, resized.cols);
            float preprocess_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            float total_ms      = preprocess_ms + result.inference_ms;
            float display_fps = (total_ms > 0.0f) ? 1000.0f/total_ms : 0.0f;

            result.preprocess_ms = preprocess_ms;

            cv::Mat mask = postprocess_mask(result.raw_output, resized.rows, resized.cols, kTraversabilityThreshold);
            cv::Mat overlay = overlay_mask(resized, mask,  kOverlayOpacity);

            // std::ostringstream oss;
            // oss << std::fixed <<std::setprecision(1)
            //     << "FPS: "<< display_fps
            //     << "  inf: " << result.inference_ms << "ms"
            //     << "  pre: " <<preprocess_ms << "ms";

            // cv::putText(overlay, oss.str(), cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 1, {255, 255, 255}, 2);
            publisher.publish(result, preprocess_ms, display_fps, overlay);
            writer.write(overlay);
            // cv::imshow("Offroad Segmentation", overlay);
            // if (cv::waitKey(1) == 27) break;
        }

        }

    catch (const std::exception& e){
        std::cerr << "Fatal error: "<< e.what()<<"\n";
        return 1;
    }

    return 0;


    }






