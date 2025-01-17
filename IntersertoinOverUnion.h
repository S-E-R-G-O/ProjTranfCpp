#ifndef INTERSECTION_H
#define INTERSECTION_H

#include <vector>  
#include <opencv2/core.hpp>  
#include <iostream>
#include <algorithm>
#include <numeric>
#include <limits>

using namespace std; // Использование стандартного пространства имен для упрощения кода

// Forward Declaration для класса TrackingBox
// Это нужно, чтобы указать компилятору, что такой класс существует, без полного определения
class TrackingBox;

class IntersectOverUnion
{
private:
    double iou_threshold_;

public:
    // Конструктор класса HungarianMethod, который принимает пороговое значение для IoU (Intersection over Union)
    IntersectOverUnion(double iou_threshold = 0.3f);

    // Метод match, который принимает матрицу IoU и выходные данные треккеров и детекций
    // Возвращает кортеж из вектора соответствий, вектора трекеров без соответствия и вектора детекций без соответствия 
    tuple<std::vector<std::vector<int>>, std::vector<int>, std::vector<int>> match(const cv::Mat& IoU, const std::vector<TrackingBox>& trackers, const std::vector<TrackingBox>& detections);

    // Статический метод для вычисления метрики IoU между двумя прямоугольниками
    static double intersectionOverUnion(const cv::Rect& a, const cv::Rect& b);


};

#endif // INTERSECTION_H
