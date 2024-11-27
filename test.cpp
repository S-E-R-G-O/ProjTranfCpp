#include "pch.h"

#include <opencv2/opencv.hpp>
#include "C:/Users/Main/source/repos/Project4/Project4/video_utils.cpp"


//���� �������� �������� �����-������1(������ ������� true ��� ��� ���� ������ ���������)
TEST(VideoUtilsTests, TestOpenVideoCaptureSuccess) {
    cv::VideoCapture cap;
    std::string validPath = "C:/Users/Main/Desktop/123/video1.avi"; 
    EXPECT_TRUE(openVideoCapture(validPath, cap));
    cap.release();
}

//���� �������� �������� �����-������2(������ ������ True ��� ��� � ���� ���� ������)
TEST(VideoUtilsTests, TestOpenVideoCaptureSuccess1) {
    cv::VideoCapture cap;
    std::string validPath = "C:/Users/Main/Desktop/123/video1.av"; 
    EXPECT_FALSE(openVideoCapture(validPath, cap));
    cap.release();
}

// ����� ��� ������� readFrame 
// (������ ������� true, ���� ���� ������� ��������)
TEST(VideoUtilsTests, TestReadFrame) {
    cv::VideoCapture cap;
    std::string validPath = "C:/Users/Main/Desktop/123/video1.avi";
    ASSERT_TRUE(openVideoCapture(validPath, cap)); // ������� ��������� �����

    cv::Mat frame;
    EXPECT_TRUE(readFrame(cap, frame)); // �������, ��� ������ ����� ����� ��������
    cap.release();
}
//����� ��� ������� readFrame
// (������ ������� false, ����� ��� ����� ����� ��������� � ��������� ����� �����)
TEST(VideoUtilsTests, TestReadFrameEndOfVideo) {
    cv::VideoCapture cap;
    std::string validPath = "C:/Users/Main/Desktop/123/video1.avi";
    ASSERT_TRUE(openVideoCapture(validPath, cap)); // ��������� �����

    cv::Mat frame;
    while (readFrame(cap, frame)); 
    EXPECT_FALSE(readFrame(cap, frame)); 
    cap.release();
}


// ���� �������� ����������� �����
// ������ ������ ������ ����(������ ������� True - ��������� ���� �� ������)
TEST(VideoUtilsTest, ShowFrame_ValidFrame) {
    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3); 
    bool isShown = showFrame(frame);
    EXPECT_TRUE(isShown);
}

// ���� �������� ����������� ������� �����
// ������ ����(������ ������� false, ��������� ���� ������)
TEST(VideoUtilsTest, ShowFrame_EmptyFrame) {
    cv::Mat emptyFrame; 
    bool isShown = showFrame(emptyFrame);
    EXPECT_FALSE(isShown);
}

