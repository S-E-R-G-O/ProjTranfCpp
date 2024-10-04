#ifndef PROCESSING_H
#define PROCESSING_H

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Класс для обработки видеопотоков
class Processing
{
public:
    // Конструктор, который инициализирует видеопотоки из указанных имен файлов
    Processing(const string& fileName1, const string& fileName2);

    // Деструктор
    ~Processing();

    // Метод для обнаружения изменений в видеопотоках
    void detectedChanges();

private:
    VideoCapture stream1; // Видеопоток 1
    VideoCapture stream2; // Видеопоток 2
    Mat previousFrame;    // Переменная для сохранения предыдущего кадра
    Mat background;       // Фон, используемый для обработки
    bool isFrame;         // Флаг, указывающий, был ли обработан первый кадр

    // Метод для обработки текущего кадра
    void processingFrame(Mat& frame, Mat& grayFrame);
};

#endif // PROCESSING_H
