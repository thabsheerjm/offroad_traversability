#pragma once
#include <string>
#include <zmq.hpp>
#include "inference.h"

class ZMQPub{
    public:
        explicit ZMQPub(const std::string& endpoint);
        void publish(const InferenceResult& result, float preprocess_ms, float display_fps, const cv::Mat& overlay);

    private:
        zmq::context_t ctx_;
        zmq::socket_t socket_;
};
