#ifndef PROCESSING_H
#define PROCESSING_H

#include "TrackingBox.h"   
#include <opencv2/opencv.hpp> 
#include <iostream> 


// Класс для обработки видео и обнаружения изменений между кадрами
class Processing
{
private:
    cv::VideoCapture stream1; // Первый видеопоток
    cv::VideoCapture stream2; // Второй видеопоток
    cv::Mat previousFrame;    // Матрица для хранения предыдущего кадра
    //cv::Mat background;
    bool isFrame;         // Булевая переменная для отслеживания наличия кадра

    // Метод для обработки одного кадра и получения объектов отслеживания

public:
    // Конструктор, принимающий имена файлов видео потоков
    Processing(const std::string& fileName1, const std::string& fileName2);

    // Деструктор
    ~Processing();

    // Метод для обнаружения изменений между двумя кадрами и возвращающий вектор отслеживаемых объектов
    std::vector<TrackingBox> detectedChanges(cv::Mat&, cv::Mat&);
    std::vector<TrackingBox> processingFrame(cv::Mat& frame, cv::Mat& grayFrame);
    cv::Mat background;

    
};

#endif // PROCESSING_H
