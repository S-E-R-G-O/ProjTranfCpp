#include "Processing.h"
#include "IntersertoinOverUnion.h"

// Конструктор класса Processing
Processing::Processing(const std::string& fileName1, const std::string& fileName2)
    : isFrame(true), background(cv::Mat())
{
    // Открываем два потока: один для каждого видео
    if (!stream1.open(fileName1) || !stream2.open(fileName2))
    {
        throw std::runtime_error("Не удалось открыть файлы: " + fileName1 + " и " + fileName2);
    }
}

// Деструктор класса Processing
Processing::~Processing()
{
    // Освобождаем ресурсы потоков видео
    stream1.release();
    stream2.release();
}

// Метод для обнаружения изменений между кадрами
std::vector<TrackingBox> Processing::detectedChanges(cv::Mat& frame, cv::Mat& thresh)
{
    cv::Mat frame1, frame2, grayFrame1;

    // Читаем текущие кадры из потоков
    stream1 >> frame1;
    stream2 >> frame2;

    // Проверяем, были ли успешно считаны кадры
    if (frame1.empty() || frame2.empty())
    {
        throw std::runtime_error("Кадры пустые; проверьте входные данные.");
    }

    // Объединяем два кадра в один
    cv::hconcat(frame1, frame2, frame);

    // Конвертируем объединенный кадр в оттенки серого
    cv::cvtColor(frame, grayFrame1, cv::COLOR_BGR2GRAY);

    // Обрабатываем текущий кадр и получаем вектор обнаруженных изменений
    std::vector<TrackingBox> det = processingFrame(frame, grayFrame1);
    grayFrame1.copyTo(thresh);

    return det;
}

// Метод для обработки каждого кадра
std::vector<TrackingBox> Processing::processingFrame(cv::Mat& frame, cv::Mat& grayFrame)
{
    cv::Mat difference, thresh, dilated, frameBlur;

    // Проверяем, инициализировано ли фоновое изображение
    if (background.empty())
    {
        background.create(grayFrame.size(), CV_32F);
        grayFrame.convertTo(background, CV_32F);
    }
    else
    {
        cv::accumulateWeighted(grayFrame, background, 0.15);
    }

    cv::Mat backgroundupd;
    background.convertTo(backgroundupd, CV_8U);
    cv::absdiff(backgroundupd, grayFrame, difference);

    cv::threshold(difference, thresh, 30, 255, cv::THRESH_BINARY);
    cv::dilate(thresh, dilated, cv::Mat(), cv::Point(-1, -1), 4);
    cv::GaussianBlur(dilated, frameBlur, cv::Size(5, 5), 0, 0);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(frameBlur, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Создаем боксы на основе обнаруженных контуров
    std::vector<TrackingBox> boxes = TrackingBox::createBoxes(contours);

    thresh.copyTo(grayFrame);

    return boxes;
}
