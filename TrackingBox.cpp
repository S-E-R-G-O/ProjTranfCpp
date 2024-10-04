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

void TrackingBox::processHistogram(const cv::Mat& frame) const {
    // Получаем координаты и размеры бокса
    auto [x, y, w, h] = this->shape();

   
    if (y + h >= frame.rows || y < 0) {
        return; 
    }


    // Извлекаем область изображения
    cv::Rect roi_top(x, y, w, h / 2);
    cv::Rect roi_bot(x, y + h / 2, w, h / 2);


    cv::Mat region_top = frame(roi_top);
    cv::Mat region_bot = frame(roi_bot);

    auto [b_hist_top, g_hist_top, r_hist_top] = computeHist(region_top);
    auto [b_hist_bot, g_hist_bot, r_hist_bot] = computeHist(region_bot);

    printMeanval(b_hist_top, g_hist_top, r_hist_top, "upper"); 
    printMeanval(b_hist_bot, g_hist_bot, r_hist_bot, "lower");


}

std::tuple<cv::Mat, cv::Mat, cv::Mat> TrackingBox::computeHist(const cv::Mat& region)
{
    std::vector<cv::Mat> bgr_planes;
    cv::split(region, bgr_planes);

    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };

    cv::Mat b_hist, g_hist, r_hist;

    // Вычисляем гистограммы для каждого канала
    cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, true, false); // Blue
    cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, true, false); // Green
    cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, true, false); // Red

    // Нормализуем гистограммы
    cv::normalize(b_hist, b_hist, 0, 1, cv::NORM_MINMAX);
    cv::normalize(g_hist, g_hist, 0, 1, cv::NORM_MINMAX);
    cv::normalize(r_hist, r_hist, 0, 1, cv::NORM_MINMAX);

    return { b_hist, g_hist, r_hist };
}

void TrackingBox::printMeanval(const cv::Mat& b_hist, const cv::Mat& g_hist, const cv::Mat& r_hist, const std::string& half)
{
    double b_mean = cv::mean(b_hist)[0];
    double g_mean = cv::mean(g_hist)[0];
    double r_mean = cv::mean(r_hist)[0];


    std::cout << "Mean color values in the " << half << " half of the tracking box: ";
    std::cout << "Blue: " << b_mean << ", Green: " << g_mean << ", Red: " << r_mean << std::endl;
}

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
