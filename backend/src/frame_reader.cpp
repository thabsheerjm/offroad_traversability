#include "frame_reader.h"
#include <stdexcept>

FrameReader::FrameReader(const std::string& source){

    if (source.size()==1 && std::isdigit(source[0])){
        cap_.open(source[0] - '0');
    }else {
        cap_.open(source);
    }

    if (!cap_.isOpened()){
        throw std::runtime_error("FrameReader: could not open source: "+ source);
    }

}

FrameReader::~FrameReader(){
    cap_.release();
}

bool FrameReader::next(cv::Mat& frame){
    // return cap_.read(frame) && !frame.empty();

    // loop back to back
    if (!cap_.read(frame) || frame.empty()){
        cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
        cap_.read(frame);
    }
    return !frame.empty();

}

int FrameReader::width() const { return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH)); }
int FrameReader::height() const { return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));}
int FrameReader::fps() const {return static_cast<int>(cap_.get(cv::CAP_PROP_FPS));}
bool FrameReader::isOpen() const {return cap_.isOpened();}
