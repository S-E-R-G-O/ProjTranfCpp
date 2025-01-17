#include "IntersertoinOverUnion.h"
#include "HungarianAlgorithm.h"
#include "TrackingBox.h"


// Функция, конвертирующая матрицу OpenCV в вектор векторов
std::vector<std::vector<double>> matToVector(const cv::Mat&);

// Конструктор класса IntersectOverUnion, инициализирующий порог IoU
IntersectOverUnion::IntersectOverUnion(double iou_threshold) : iou_threshold_(iou_threshold) {}

// Метод для вычисления IoU между двумя прямоугольниками
double IntersectOverUnion::intersectionOverUnion(const cv::Rect& a, const cv::Rect& b) {
    // Находим координаты пересечения
    int xA = std::max(a.x, b.x);
    int yA = std::max(a.y, b.y);
    int xB = std::min(a.x + a.width, b.x + b.width);
    int yB = std::min(a.y + a.height, b.y + b.height);

    // Вычисляем площадь пересечения
    int interArea = std::max(0, xB - xA) * std::max(0, yB - yA);
    // Вычисляем площади прямоугольников a и b
    double boxAArea = a.width * a.height;
    double boxBArea = b.width * b.height;

    // Вычисляем IoU
    return interArea / (boxAArea + boxBArea - interArea);
}

// Метод для сопоставления трекеров и обнаруженных объектов на основе матрицы IoU
std::tuple<std::vector<std::vector<int>>, std::vector<int>, std::vector<int>> IntersectOverUnion::match(const cv::Mat& IoU, const std::vector<TrackingBox>& trackers, const std::vector<TrackingBox>& detections) {
    std::vector<int> assignment; // Массив для сопоставления
    std::vector<std::vector<double>> costMatrix = matToVector(1 - IoU); // Инвертируем IoU, т.к. венгерский алгоритм находит минимальные стоимости

    // Создаем экземпляр венгерского алгоритма
    HungarianAlgorithm hungAlgo;

    // Выполняем венгерский алгоритм и получаем стоимость сопоставления
    double cost = hungAlgo.Solve(costMatrix, assignment);

    std::vector<std::vector<int>> matches; // Вектор для хранения сопоставлений
    std::vector<int> unmatched_trackers, unmatched_detections; // Векторы для хранения несопоставленных объектов

    // Поиск сопоставлений по матрице IoU
    for (size_t t = 0; t < trackers.size(); ++t) {
        if (assignment[t] == -1 || IoU.at<double>(t, assignment[t]) < iou_threshold_) {
            unmatched_trackers.push_back(t); // Если нет, добавляем в несопоставленные трекеры
        } else {
            matches.push_back({ static_cast<int>(t), static_cast<int>(assignment[t]) }); // Добавляем корректные сопоставления
        }
    }

    // Поиск несопоставленных детекций
    for (size_t d = 0; d < detections.size(); ++d) {
        if (std::none_of(matches.begin(), matches.end(), [d](const std::vector<int>& match) { return match[1] == d; })) {
            unmatched_detections.push_back(d); // Если нет, добавляем в несопоставленные детекции
        }
    }

    // Возвращаем найденные сопоставления и несопоставленные объекты
    return { matches, unmatched_detections, unmatched_trackers };
}

// Функция для конвертации матрицы OpenCV в вектор векторов
std::vector<std::vector<double>> matToVector(const cv::Mat& mat) {
    if (mat.empty()) {
        return {};
    }

    std::vector<std::vector<double>> vec(mat.rows, std::vector<double>(mat.cols));

    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            vec[i][j] = mat.at<double>(i, j);
        }
    }

    return vec; // Возвращаем результат
}
