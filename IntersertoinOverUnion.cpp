#include "IntersertoinOverUnion.h" 
#include "HungarianAlgorithm.h" 
#include "TrackingBox.h"
#include <iostream> 
#include <algorithm> 
#include <numeric> 
#include <limits> 



// Функция, конвертирующая матрицу OpenCV в вектор векторов
std::vector<std::vector<double>> matToVector(const cv::Mat&);

// Конструктор класса HungarianMethod, инициализирующий порог IoU
IntersectOverUnion::IntersectOverUnion(double iou_threshold) : iou_threshold_(iou_threshold) {}

// Метод для вычисления IoU между двумя прямоугольниками
double IntersectOverUnion::intersectionOverUnion(const cv::Rect& a, const cv::Rect& b) {
    // Находим координаты пересечения
    int xA = max(a.x, b.x);
    int yA = max(a.y, b.y);
    int xB = min(a.x + a.width, b.x + b.width);
    int yB = min(a.y + a.height, b.y + b.height);

    // Вычисляем площадь пересечения
    int interArea = max(0, xB - xA) * max(0, yB - yA);
    // Вычисляем площади прямоугольников a и b
    double boxAArea = a.width * a.height;
    double boxBArea = b.width * b.height;

    // Вычисляем IoU
    return interArea / (boxAArea + boxBArea - interArea);
}

// Метод для сопоставления трекеров и обнаруженных объектов на основе матрицы IoU
std::tuple<std::vector<std::vector<int>>, std::vector<int>, std::vector<int>> IntersectOverUnion::match(const cv::Mat& IoU, const std::vector<TrackingBox>& trackers, const std::vector<TrackingBox>& detections) {
    vector<int> assignment; // Массив для сопоставления

    // Конвертация матрицы IoU из OpenCV в вектор векторов для венгерского алгоритма
    vector<vector<double>> costMatrix = matToVector(1 - IoU); // Инвертируем IoU, т.к. венгерский алгоритм находит минимальные стоимости

    // Создаем экземпляр венгерского алгоритма
    HungarianAlgorithm HungAlgo;

    // Выполняем венгерский алгоритм и получаем стоимость сопоставления
    double cost = HungAlgo.Solve(costMatrix, assignment);
    //Тесты
    // Выводим информацию о сопоставлении
    /*
    std::cout << "assignment: " << assignment.size() << std::endl;
    for (unsigned int x = 0; x < costMatrix.size(); x++)
        std::cout << x << " - " << assignment[x] << "\t";
        */
    vector<vector<int>> matches; // Вектор для хранения сопоставлений
    vector<int> unmatched_trackers, unmatched_detections; // Векторы для хранения несопоставленных объектов

    // Поиск сопоставлений по матрице IoU
    for (size_t t = 0; t < trackers.size(); ++t) {
        // Проверяем, есть ли сопоставление и превышает ли оно порог IoU
        if (assignment[t] == -1 || IoU.at<double>(t, assignment[t]) < iou_threshold_) {
            unmatched_trackers.push_back(t); // Если нет, добавляем в несопоставленные трекеры
        }
        else {
            matches.push_back({ static_cast<int>(t), static_cast<int>(assignment[t]) }); // Добавляем корректные сопоставления
        }
    }

    // Поиск несопоставленных детекций
    for (size_t d = 0; d < detections.size(); ++d) {
        bool matched = true;
        for (unsigned int x = 0; x < matches.size(); x++)
        {
            if (matches[x][1] == d) matched = false; // Проверяем, сопоставлена ли детекция
        }
        if (matched) {
            unmatched_detections.push_back(d); // Если нет, добавляем в несопоставленные детекции
        }
    }
    //Тесты
    // Выводим информацию о найденных сопоставлениях и несопоставленных объектах
    /*
    std::cout << "\n----------" << std::endl;
    std::cout << "matches: " << matches.size() << std::endl;
    for (unsigned int x = 0; x < matches.size(); x++)
        std::cout << matches[x][0] << "<-->" << matches[x][1] << "\t";
    std::cout << "\nunmatched_trackers: " << unmatched_trackers.size() << std::endl;
    for (unsigned int x = 0; x < unmatched_trackers.size(); x++)
        std::cout << unmatched_trackers[x] << "\t";
    std::cout << "\nunmatched_detections: " << unmatched_detections.size() << std::endl;
    for (unsigned int x = 0; x < unmatched_detections.size(); x++)
        std::cout << unmatched_detections[x] << "\t";
    std::cout << "\n--------END---------" << std::endl;
    */
    // Возвращаем найденные сопоставления и несопоставленные объекты
    return { matches, unmatched_detections, unmatched_trackers };
}

// Функция для конвертации матрицы OpenCV в вектор векторов
std::vector < std::vector<double> > matToVector(const cv::Mat& mat) {
    // Проверяем, не пустая ли матрица
    if (mat.empty()) {
        return {};
    }

    // Создаем вектор векторов для хранения данных
    std::vector < std::vector <double> > vec(mat.rows, std::vector<double>(mat.cols));

    // Копируем данные из cv::Mat в std::vector<std::vector<double>>
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            vec[i][j] = mat.at<double>(i, j);
        }
    }

    return vec; // Возвращаем результат
}
