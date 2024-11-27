// video_utils.h
#ifndef VIDEO_UTILS_H
#define VIDEO_UTILS_H

#include <opencv2/opencv.hpp>
#include <string>

bool openVideoCapture(const std::string& videoPath, cv::VideoCapture& cap);
bool readFrame(cv::VideoCapture& cap, cv::Mat& frame);
bool showFrame(const cv::Mat& frame);

#endif // VIDEO_UTILS_H
