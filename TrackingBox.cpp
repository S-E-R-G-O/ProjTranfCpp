#include "TrackingBox.h"

// Определение статических переменных
cv::Scalar TrackingBox::GreenColor = cv::Scalar(100, 255, 0); // Зеленый цвет
cv::Scalar TrackingBox::RedColor = cv::Scalar(0, 0, 255); // Красный цвет

// Конструктор, принимающий координаты (x, y) и размеры (w, h) прямоугольника
TrackingBox::TrackingBox(int x, int y, int w, int h) : x(x), y(y), w(w), h(h) {}

// Метод для получения формы (координаты и размеры) прямоугольника в виде кортежа
std::tuple<int, int, int, int> TrackingBox::shape() const {
    return std::make_tuple(x, y, w, h);
}

// Метод для получения координат прямоугольника в виде вектора
std::vector<int> TrackingBox::rectangle() const {
    return { x, y, x + w, y + h }; // Возвращает вектор, содержащий координаты левого верхнего и правого нижнего углов
}

// Метод для создания массива TrackingBox из контуров
std::vector<TrackingBox> TrackingBox::createBoxes(const std::vector<std::vector<cv::Point>>& contours) {
    std::vector<TrackingBox> boxes; // Вектор для хранения созданных боксов

    for (const auto& cnt : contours) {
        // Проверка площади контура на минимальное значение
        if (cv::contourArea(cnt) > det_limArea) {
            cv::Rect rect = cv::boundingRect(cnt); // Получаем ограничивающий прямоугольник для контура
            // Добавляем новый TrackingBox на основе ограничивающего прямоугольника
            boxes.emplace_back(rect.x, rect.y, rect.width, rect.height);
        }
    }
    return boxes; // Возвращаем массив созданных боксов
}

// Метод для создания гистограммы из областей прямоугольников


// Метод для рисования боксов на изображении
cv::Mat TrackingBox::drawBoxes(cv::Mat& frame, const std::vector<TrackingBox>& boxes) {
    for (const auto& box : boxes) {
        auto [x, y, w, h] = box.shape(); // Получаем координаты и размеры бокса
        int cx = x + w / 2; // Находим центр бокса по оси X
        int cy = y + h / 2; // Находим центр бокса по оси Y

        // Рисуем прямоугольник на изображении
        cv::rectangle(frame, cv::Point(x, y), cv::Point(x + w, y + h), GreenColor, 2);
        // Рисуем горизонтальную линию в центре бокса
        cv::line(frame, cv::Point(x, cy), cv::Point(x + w, cy), RedColor, 2);
    }
    return frame; // Возвращаем модифицированное изображение
}
