// source.cpp
#include <opencv2/opencv.hpp>
#include <iostream>
#include "video_utils.h"

int main() {
    cv::VideoCapture cap;
    const std::string videoPath = "C:/Users/Main/Desktop/123/video1.avi"; // Укажите путь к вашему AVI файлу

    if (!openVideoCapture(videoPath, cap)) {
        std::cerr << "Ошибка: Не удалось открыть видеофайл." << std::endl;
        return -1;
    }

    cv::Mat frame;

    while (true) {
        if (!readFrame(cap, frame)) {
            std::cout << "Кадры закончились." << std::endl;
            break;
        }

        if (!showFrame(frame)) {
            std::cerr << "Ошибка: Не удалось отобразить кадр." << std::endl;
        }

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
