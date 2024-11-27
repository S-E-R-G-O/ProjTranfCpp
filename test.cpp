#include "pch.h"

#include <opencv2/opencv.hpp>
#include "C:/Users/Main/source/repos/Project4/Project4/video_utils.cpp"


//Тест проверки открытия видео-потока1(Должен вернуть true так как путь указан правильно)
TEST(VideoUtilsTests, TestOpenVideoCaptureSuccess) {
    cv::VideoCapture cap;
    std::string validPath = "C:/Users/Main/Desktop/123/video1.avi"; 
    EXPECT_TRUE(openVideoCapture(validPath, cap));
    cap.release();
}

//Тест проверки открытия видео-потока2(Должен выдать True так как в пути есть ошибка)
TEST(VideoUtilsTests, TestOpenVideoCaptureSuccess1) {
    cv::VideoCapture cap;
    std::string validPath = "C:/Users/Main/Desktop/123/video1.av"; 
    EXPECT_FALSE(openVideoCapture(validPath, cap));
    cap.release();
}

// Тесты для функции readFrame 
// (Должен вернуть true, если кадр успешно прочитан)
TEST(VideoUtilsTests, TestReadFrame) {
    cv::VideoCapture cap;
    std::string validPath = "C:/Users/Main/Desktop/123/video1.avi";
    ASSERT_TRUE(openVideoCapture(validPath, cap)); // Сначала открываем видео

    cv::Mat frame;
    EXPECT_TRUE(readFrame(cap, frame)); // Ожидаем, что чтение кадра будет успешным
    cap.release();
}
//Тесты для функции readFrame
// (Должен вернуть false, когда все кадры будут прочитаны и достигнут конец видео)
TEST(VideoUtilsTests, TestReadFrameEndOfVideo) {
    cv::VideoCapture cap;
    std::string validPath = "C:/Users/Main/Desktop/123/video1.avi";
    ASSERT_TRUE(openVideoCapture(validPath, cap)); // Открываем видео

    cv::Mat frame;
    while (readFrame(cap, frame)); 
    EXPECT_FALSE(readFrame(cap, frame)); 
    cap.release();
}


// Тест проверки отображения кадра
// Создаём пустой черный кадр(Должен вернуть True - поскольку кадр не пустой)
TEST(VideoUtilsTest, ShowFrame_ValidFrame) {
    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3); 
    bool isShown = showFrame(frame);
    EXPECT_TRUE(isShown);
}

// Тест проверки отображения пустого кадра
// Пустой кадр(Должен вернуть false, поскольку кадр пустой)
TEST(VideoUtilsTest, ShowFrame_EmptyFrame) {
    cv::Mat emptyFrame; 
    bool isShown = showFrame(emptyFrame);
    EXPECT_FALSE(isShown);
}

