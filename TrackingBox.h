#pragma once
#ifndef TRK_BOX
#define TRK_BOX

#include <opencv2/opencv.hpp> // Подключение библиотеки OpenCV для работы с изображениями
#include <vector> // Подключение вектора для работы с динамическими массивами
#include <tuple> // Для использования std::tuple
#include <string> // Для использования std::string

// Класс для представления прямоугольных областей отслеживания
class TrackingBox {
public:
    // Статические переменные для хранения цветов (зеленый и красный)
    static cv::Scalar GreenColor; // Зеленый цвет
    static cv::Scalar RedColor;    // Красный цвет
    static const int det_limArea = 6000; // Константа для ограничения площади детекции

    // Конструктор класса, принимающий координаты и размеры прямоугольника
    TrackingBox(int x, int y, int w, int h);

    // Метод для получения формы (координаты и размеры) прямоугольника
    std::tuple<int, int, int, int> shape() const;

    // Метод для получения координат прямоугольника в виде вектора
    std::vector<int> rectangle() const;

    // Метод для обработки гистограммы из области изображения
    void processHistogram(const cv::Mat& frame) const;

    // Статический метод для создания объектов TrackingBox на основе контуров
    static std::vector<TrackingBox> createBoxes(const std::vector<std::vector<cv::Point>>& contours);

    // Статический метод для отрисовки прямоугольников на изображении
    static cv::Mat drawBoxes(cv::Mat& frame, const std::vector<TrackingBox>& boxes);

private:
    // Приватные переменные для хранения координат и размеров прямоугольника
    int x, y, w, h;

    // Приватные методы
    static std::tuple<cv::Mat, cv::Mat, cv::Mat> computeHist(const cv::Mat& region);
    static void printMeanval(const cv::Mat& b_hist, const cv::Mat& g_hist, const cv::Mat& r_hist, const std::string& half);
};

#endif // Защита от повторного включения заголовочного файла
