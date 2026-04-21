#include "zmq_pub.h"
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

ZMQPub::ZMQPub(const std::string& endpoint)
    : ctx_(1), socket_(ctx_, zmq::socket_type::pub)
{
    socket_.bind(endpoint);
}

// ASCII base64
static std::string base64_encode(const std::vector<uchar>& buf) {
    static const char t[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((buf.size() + 2) / 3) * 4);
    for (size_t i = 0; i < buf.size(); i += 3) {
        uint32_t b = buf[i] << 16;
        if (i + 1 < buf.size()) b |= buf[i+1] << 8;
        if (i + 2 < buf.size()) b |= buf[i+2];
        out += t[(b >> 18) & 0x3f];
        out += t[(b >> 12) & 0x3f];
        out += (i + 1 < buf.size()) ? t[(b >> 6) & 0x3f] : '=';
        out += (i + 2 < buf.size()) ? t[b & 0x3f]        : '=';
    }
    return out;
}

void ZMQPub::publish(const InferenceResult& result,
                           float preprocess_ms, float display_fps, const cv::Mat& overlay)
{
    // convert float mask -> 8-bit -> PNG bytes
    // std::vector<uchar> buf;
    // cv::Mat mask_8u;
    // result.traversability_mask.convertTo(mask_8u, CV_8U, 255.0);
    // cv::imencode(".png", mask_8u, buf);

    std::vector<uchar> overlay_buf;
    cv::imencode(".jpg", overlay, overlay_buf);


    nlohmann::json msg;
    msg["timestamp_us"]  = result.timestamp_us;
    msg["model_id"]      = result.model_id;
    msg["inference_ms"]  = result.inference_ms;
    msg["preprocess_ms"] = preprocess_ms;
    msg["display_fps"]   = display_fps;
    msg["confidence"]    = result.confidence;
    // msg["mask_png_b64"]  = base64_encode(buf);
    msg["overlay_jpg_b64"] = base64_encode(overlay_buf);

    std::string payload = msg.dump();
    socket_.send(zmq::buffer(payload), zmq::send_flags::dontwait);
}
