#pragma once
#ifndef TRK_BOX
#define TRK_BOX

#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>
#include <string>
#include <unordered_map>
#include <iostream>

// Класс TrackingBox предназначен для представления ограничивающего прямоугольника,
// в котором производится отслеживание объектов.
class TrackingBox {
private:
    int x, y, w, h;            // Координаты верхнего левого угла (x, y) и размеры (w, h) ограничивающего прямоугольника
    int id;                    // Идентификатор объекта (каждый объект отслеживания имеет уникальный ID)

    static int id_counter;     // Счетчик для уникальных идентификаторов

public:
    static cv::Scalar GreenColor;
    static cv::Scalar RedColor;
    static const int det_limArea = 6000; // Минимальная площадь для детекции

    TrackingBox(int x, int y, int w, int h);

    // Хранение обновленных ID
    static std::vector<int> updated_ids;

    // Метод для получения формы (координаты и размеры) ограничивающего прямоугольника
    std::tuple<int, int, int, int> shape() const;

    // Метод для получения прямоугольника
    cv::Rect rectangle() const;

    // Обработка гистограммы на основе текущего кадра
    void processHistogram(const cv::Mat& frame);

    // Создание ограничивающих прямоугольников на основе контуров
    static std::vector<TrackingBox> createBoxes(const std::vector<std::vector<cv::Point>>& contours);

    // Рисование ограничивающих прямоугольников на изображении
    static cv::Mat drawBoxes(cv::Mat& frame, const std::vector<TrackingBox>& boxes);

    // Создание отслеживания на основе детекций и трекеров
    static std::vector<TrackingBox> trackingCreation(std::vector<TrackingBox>& detections, std::vector<TrackingBox>& trackers);

    // Печать информации о текущем объекте
    void Print() const { std::cout << id << ":(" << x << ", " << y << ", " << w << ", " << h << ")" << std::endl; }

    // Сравнение гистограмм
    static std::unordered_map<int, std::vector<double>> compareHistograms(const cv::Mat& new_hist);

    // Печать средних значений гистограммы
    static void printMeanval(const cv::Mat& b_hist, const cv::Mat& g_hist, const cv::Mat& r_hist, const std::string& half);

    // Вычисление гистограммы для заданного региона
    static std::tuple<cv::Mat, cv::Mat, cv::Mat> computeHist(const cv::Mat& region);

    // Поиск лучшего ID на основе весов гистограммы
    static std::tuple<int, double> findBestId(const std::unordered_map<int, std::vector<double>>& hist_weights, const std::string& half);

    // Хранение историй для отслеживания
    static std::unordered_map<int, std::vector<cv::Mat>> track_hist; // История треков
    static std::unordered_map<int, std::vector<cv::Mat>> del_hists;  // Удаленные истории

    // Обновление ID и печать информации
    void updateIdAndPrint(int best_id_top, double best_weight_top, int best_id_bot, double best_weight_bot);

    // Получение текущего ID
    int getId() const { return id; }

    // Метод для распространения ID на основе похожести
    void propagateIds();

    // Новый метод для вычисления IoU между двумя прямоугольниками
    static double intersectionOverUnion(const cv::Rect& a, const cv::Rect& b);

    // Новый метод для сопоставления трекеров и обнаруженных объектов на основе матрицы IoU
    static std::tuple<std::vector<std::vector<int>>, std::vector<int>, std::vector<int>> match(const cv::Mat& IoU, const std::vector<TrackingBox>& trackers, const std::vector<TrackingBox>& detections);
};

#endif // TRK_BOX
