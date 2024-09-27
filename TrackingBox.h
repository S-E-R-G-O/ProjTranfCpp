#pragma once
#ifndef TRK_BOX
#define TRK_BOX

#include <opencv2/opencv.hpp> // Подключение библиотеки OpenCV для работы с изображениями
#include <vector> // Подключение вектора для работы с динамическими массивами

// Класс для представления прямоугольных областей отслеживания
class TrackingBox {
public:
    // Статические переменные для хранения цветов (зеленый и красный)
    static cv::Scalar GreenColor; // Зеленый цвет
    static cv::Scalar RedColor;    // Красный цвет
    static const int det_limArea = 6000; // Константа для ограничения площади детекции

    // Конструктор класса, принимающий координаты и размеры прямоугольника
    TrackingBox(int x, int y, int w, int h);

    // Метод для получения формы (ширина и высота) прямоугольника
    std::tuple<int, int, int, int> shape() const;

    // Метод для получения координат прямоугольника в виде вектора
    std::vector<int> rectangle() const;

    // Статический метод для создания гистограммы на основе кадра и прямоугольников


    // Статический метод для создания объектов TrackingBox на основе контуров
    static std::vector<TrackingBox> createBoxes(const std::vector<std::vector<cv::Point>>& contours);

    // Статический метод для отрисовки прямоугольников на изображении
    static cv::Mat drawBoxes(cv::Mat& frame, const std::vector<TrackingBox>& boxes);

private:
    // Приватные переменные для хранения координат и размеров прямоугольника
    int x, y, w, h;
};

#endif // Защита от повторного включения заголовочного файла
