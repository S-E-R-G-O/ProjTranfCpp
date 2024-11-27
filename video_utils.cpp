// video_utils.cpp
#include "video_utils.h"

bool openVideoCapture(const std::string& videoPath, cv::VideoCapture& cap) {
    cap.open(videoPath);
    return cap.isOpened();
}

bool readFrame(cv::VideoCapture& cap, cv::Mat& frame) {
    cap >> frame;
    return !frame.empty();
}

bool showFrame(const cv::Mat& frame) {
    if (frame.empty()) {
        return false;
    }
    cv::imshow("Video", frame);
    return true;
}
