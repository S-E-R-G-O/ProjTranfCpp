#pragma once
#ifndef TRK_BOX
#define TRK_BOX

#include <opencv2/opencv.hpp> 
#include <vector>            
#include <tuple>             
#include <string>            
#include <unordered_map>    
#include <iostream> // Добавлено для вывода отладочной информации

// Класс TrackingBox предназначен для представления ограничивающего прямоугольника,
// в котором производится отслеживание объектов.
class TrackingBox {
   
public:
    // Статическая переменная для цвета зелёного цвета (например, для отрисовки)
    static cv::Scalar GreenColor;
    // Статическая переменная для цвета красного цвета
    static cv::Scalar RedColor;
    // Константа для ограничения минимальной площади объекта
    static const int det_limArea = 6000;

    // Конструктор класса, принимающий координаты и размеры ограничивающего прямоугольника
    TrackingBox(int x, int y, int w, int h);
    // Метод для получения параметров формы (x, y, w, h) в виде кортежа
    std::tuple<int, int, int, int> shape() const;
    // Метод для получения ограничивающего прямоугольника OpenCV
    cv::Rect rectangle() const;

    // Метод для обработки гистограммы из предоставленного кадра
    void processHistogram(const cv::Mat& frame);

    // Метод для создания множества TrackingBox на основе полученных контуров
    static std::vector<TrackingBox> createBoxes(const std::vector<std::vector<cv::Point>>& contours);

    // Метод для отрисовки ограничивающих прямоугольников на кадре
    static cv::Mat drawBoxes(cv::Mat& frame, const std::vector<TrackingBox>& boxes);

    // Метод для создания трекеров, используя детекции и существующие трекеры
    static std::vector<TrackingBox> trackingCreation(std::vector<TrackingBox>& detections, std::vector<TrackingBox>& trackers);

    // Метод для вывода информации об объекте в консоль
    void Print() const { std::cout << id << ":(" << x << ", " << y << ", " << w << ", " << h << ")" << std::endl; }

    // Метод для сравнения гистограмм
    static std::unordered_map<int, std::vector<double>> compareHistograms(const cv::Mat& new_hist);
    // Метод для вывода средних значений гистограммы
    static void printMeanval(const cv::Mat& b_hist, const cv::Mat& g_hist, const cv::Mat& r_hist, const std::string& half);
    static std::tuple<cv::Mat, cv::Mat, cv::Mat> computeHist(const cv::Mat& region);
    static std::unordered_map<int, std::vector<cv::Mat>> track_hist;
    static std::unordered_map<int, std::vector<cv::Mat>> del_hists;
    int getId() const { return id; }
   
private:
    int x, y, w, h;            // Координаты верхнего левого угла (x, y) и размеры (w, h) ограничивающего прямоугольника
    int id;                    // Идентификатор объекта (каждый объект отслеживания имеет уникальный ID

    static int id_counter;
    
};

#endif // TRK_BOX
