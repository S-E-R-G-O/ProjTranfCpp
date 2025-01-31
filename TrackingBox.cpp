#include <unordered_map>
#include <iostream>
#include <numeric>
#include <vector>
#include "TrackingBox.h"
#include "HungarianAlgorithm.h"

// Инициализация статических переменных класса TrackingBox
cv::Scalar TrackingBox::GreenColor = cv::Scalar(100, 255, 0); // Зеленый цвет для отрисовки
cv::Scalar TrackingBox::RedColor = cv::Scalar(0, 0, 255); // Красный цвет для отрисовки
std::unordered_map<int, std::vector<cv::Mat>> TrackingBox::track_hist; // Хранение историй трекеров
std::unordered_map<int, std::vector<cv::Mat>> TrackingBox::del_hists; // Хранение удаленных историй
int TrackingBox::id_counter = 0; // Счетчик ID для трекеров
std::vector<int> TrackingBox::updated_ids; // Список обновленных ID
std::unordered_map<int, std::vector<int>> similar_ids; // Схожие ID

// Конструктор класса TrackingBox
TrackingBox::TrackingBox(int x, int y, int w, int h)
    : x(x), y(y), w(w), h(h), id(id_counter++) {} // Инициализация координат и присвоение уникального ID

// Метод для создания трекеров на основе обнаруженных объектов
std::vector<TrackingBox> TrackingBox::trackingCreation(
    std::vector<TrackingBox>& detections,
    std::vector<TrackingBox>& trackers) {

    if (trackers.empty()) return detections; // Если трекеров нет, вернуть обнаруженные

    if (detections.empty()) { // Если нет обнаруженных объектов
        for (auto& t : trackers) { // Удаляем трекеры
            del_hists[t.id] = std::move(track_hist[t.id]);
            track_hist.erase(t.id);
            std::cout << "DEL " << t.id << " " << del_hists[t.id].size() << std::endl; // Выводим информацию об удалении
        }
        trackers.clear(); // Очищаем список трекеров
        return {};
    }

    cv::Mat IoU(trackers.size(), detections.size(), CV_64F); // Создаем матрицу IoU
    for (size_t t = 0; t < trackers.size(); ++t) {
        for (size_t d = 0; d < detections.size(); ++d) {
            IoU.at<double>(t, d) = TrackingBox::intersectionOverUnion(trackers[t].rectangle(), detections[d].rectangle()); // Заполняем матрицу IoU
        }
    }

    auto [matches, unmatched_det_ids, unmatched_trk_ids] = match(IoU, trackers, detections); // Выполняем сопоставление

    for (const auto& mtcd : matches) { // Обновляем ID для сопоставленных объектов
        detections[mtcd[1]].id = trackers[mtcd[0]].id;
        trackers[mtcd[0]] = detections[mtcd[1]]; // Обновляем трекеры
    }

    for (auto det_id : unmatched_det_ids) { // Добавляем несопоставленные обнаруженные объекты в трекеры
        trackers.push_back(detections[det_id]);
    }

    for (auto ut = unmatched_trk_ids.rbegin(); ut != unmatched_trk_ids.rend(); ++ut) { // Удаляем несопоставленные трекеры
        del_hists[trackers[*ut].id] = std::move(track_hist[trackers[*ut].id]);
        track_hist.erase(trackers[*ut].id);
        std::cout << "DEL " << trackers[*ut].id << " " << del_hists[trackers[*ut].id].size() << std::endl;
        trackers.erase(trackers.begin() + *ut);
    }

    return trackers; // Возвращаем обновленный список трекеров
}

// Метод для получения формы трекера
std::tuple<int, int, int, int> TrackingBox::shape() const {
    return std::make_tuple(x, y, w, h); // Возвращаем координаты и размеры
}

// Метод для получения прямоугольника трекера
cv::Rect TrackingBox::rectangle() const {
    return cv::Rect(x, y, w, h); // Возвращаем прямоугольник
}

// Метод для создания трекеров на основе контуров
std::vector<TrackingBox> TrackingBox::createBoxes(const std::vector<std::vector<cv::Point>>& contours) {
    std::vector<TrackingBox> boxes;

    for (const auto& cnt : contours) {
        if (cv::contourArea(cnt) > det_limArea) { // Проверяем площадь контура
            cv::Rect rect = cv::boundingRect(cnt); // Получаем ограничивающий прямоугольник
            boxes.emplace_back(rect.x, rect.y, rect.width, rect.height); // Создаем новый трекер
        }
    }
    return boxes; // Возвращаем созданные трекеры
}

// Метод для нахождения лучшего ID на основе гистограмм
std::tuple<int, double> TrackingBox::findBestId(const std::unordered_map<int, std::vector<double>>& hist_weights, const std::string& half) {
    int best_id = -1;
    double best_weight = -1.0;

    for (const auto& [hw_id, weights] : hist_weights) {
        double mean_value = std::accumulate(weights.begin(), weights.end(), 0.0) / weights.size(); // Вычисляем среднее значение веса
        std::cout << half << " ГИСТОГРАММА: " << hw_id << ": " << mean_value << std::endl;

        if (mean_value > best_weight) { // Находим ID с наибольшим весом
            best_weight = mean_value;
            best_id = hw_id;
        }
    }

    return { best_id, best_weight }; // Возвращаем лучший ID и его вес
}

// Метод для обновления ID и вывода информации
void TrackingBox::updateIdAndPrint(int best_id_top, double best_weight_top, int best_id_bot, double best_weight_bot) {
    if (best_id_top != -1) {
        if (best_id_top == best_id_bot) {
            std::cout << "Обновление ID: " << this->id << " на " << best_id_top << std::endl;
            this->id = best_id_top; // Обновляем ID
            if (std::find(updated_ids.begin(), updated_ids.end(), best_id_top) == updated_ids.end()) {
                updated_ids.push_back(best_id_top); // Добавляем в список обновленных ID
            }
        }
        std::cout << "Наиболее похожий объект для верхней гистограммы: " << best_id_top << " с весом: " << best_weight_top << std::endl;
        if (std::find(updated_ids.begin(), updated_ids.end(), best_id_top) == updated_ids.end()) {
            updated_ids.push_back(best_id_top);
        }
        if (this->id != best_id_top) {
            similar_ids[this->id].push_back(best_id_top); 
        }
    }

    if (best_id_bot != -1) {
        std::cout << "Наиболее похожий объект для нижней гистограммы: " << best_id_bot << " с весом: " << best_weight_bot << std::endl;
        if (std::find(updated_ids.begin(), updated_ids.end(), best_id_bot) == updated_ids.end()) {
            updated_ids.push_back(best_id_bot);
        }
        if (this->id != best_id_bot) {
            similar_ids[this->id].push_back(best_id_bot);
        }
    }

    propagateIds(); // Распространяем ID
}

// Метод для распространения ID
void TrackingBox::propagateIds() {
    std::unordered_map<int, int> final_ids;

    for (const auto& [id, similar] : similar_ids) { // Для каждого ID находим его финальный ID
        int current_id = id;
        for (int sim_id : similar) {
            if (final_ids.find(sim_id) != final_ids.end()) {
                current_id = final_ids[sim_id]; // Если ID найден, обновляем текущий ID
            }
        }
        final_ids[id] = current_id; // Сохраняем финальный ID
    }

    if (final_ids.find(this->id) != final_ids.end()) {
        this->id = final_ids[this->id]; // Обновляем текущий ID
        std::cout << "Обновленный ID после распространения: " << this->id << std::endl;
    }
}

// Метод для обработки гистограммы
void TrackingBox::processHistogram(const cv::Mat& frame) {
    auto [x, y, w, h] = this->shape(); // Получаем форму трекера

    if (y + h >= frame.rows || y < 0) { // Проверяем границы изображения
        return;
    }

    cv::Rect roi_top(x, y, w, h / 2); // Верхняя область интереса
    cv::Rect roi_bot(x, y + h / 2, w, h / 2); // Нижняя область интереса

    cv::Mat region_top = frame(roi_top); // Получаем верхнюю область
    cv::Mat region_bot = frame(roi_bot); // Получаем нижнюю область

    auto [b_hist_top, g_hist_top, r_hist_top] = computeHist(region_top); // Вычисляем гистограммы
    auto [b_hist_bot, g_hist_bot, r_hist_bot] = computeHist(region_bot);

    auto new_hist_top = (b_hist_top + g_hist_top + r_hist_top) / 3.0; // Средняя гистограмма верхней области
    auto new_hist_bot = (b_hist_bot + g_hist_bot + r_hist_bot) / 3.0; // Средняя гистограмма нижней области

    if (track_hist.find(this->id) != track_hist.end()) { // Если ID уже есть в трекерах
        track_hist[this->id].push_back(new_hist_top); // Добавляем новые гистограммы
        track_hist[this->id].push_back(new_hist_bot);
    }
    else {
        std::cout << "NEW " << this->id << std::endl;

        auto hist_weight_top = compareHistograms(new_hist_top); // Сравниваем гистограммы
        auto hist_weight_bot = compareHistograms(new_hist_bot);

        auto [best_id_top, best_weight_top] = findBestId(hist_weight_top, "ВЕРХНЯЯ"); // Находим лучший ID для верхней части
        auto [best_id_bot, best_weight_bot] = findBestId(hist_weight_bot, "НИЖНЯЯ"); // Находим лучший ID для нижней части

        updateIdAndPrint(best_id_top, best_weight_top, best_id_bot, best_weight_bot); // Обновляем ID и выводим информацию

        track_hist[this->id].push_back(new_hist_bot); // Добавляем нижнюю гистограмму в историю
    }
}

// Метод для вычисления гистограммы
std::tuple<cv::Mat, cv::Mat, cv::Mat> TrackingBox::computeHist(const cv::Mat& region) {
    std::vector<cv::Mat> bgr_planes; // Вектор для цветовых каналов
    cv::split(region, bgr_planes); // Разделяем изображение на каналы

    cv::Mat histograms[3]; // Массив для хранения гистограмм
    int histSize = 256; // Размер гистограммы
    float range[] = { 0, 256 }; // Диапазон
    const float* histRange = { range };

    for (int i = 0; i < 3; ++i) { // Вычисляем гистограммы для каждого канала
        if (!bgr_planes[i].empty()) {
            cv::calcHist(&bgr_planes[i], 1, 0, cv::Mat(), histograms[i], 1, &histSize, &histRange, true, false);
            cv::normalize(histograms[i], histograms[i], 0, 1, cv::NORM_MINMAX); // Нормализуем гистограммы
        }
    }

    return { histograms[0], histograms[1], histograms[2] }; // Возвращаем гистограммы для каждого канала
}

// Метод для вывода среднего значения гистограммы
void TrackingBox::printMeanval(const cv::Mat& b_hist, const cv::Mat& g_hist, const cv::Mat& r_hist, const std::string& half) {
    double av_mean = (cv::mean(b_hist)[0] + cv::mean(g_hist)[0] + cv::mean(r_hist)[0]) / 3.0; // Вычисляем среднее значение
    std::cout << "Mean color values in the " << half << " half of the tracking box: Average: " << av_mean << std::endl; // Выводим среднее значение
}

// Метод для сравнения гистограмм
std::unordered_map<int, std::vector<double>> TrackingBox::compareHistograms(const cv::Mat& new_hist) {
    std::unordered_map<int, std::vector<double>> hist_weight;

    for (const auto& [id, hist_vec] : del_hists) { // Для каждой удаленной гистограммы
        for (const auto& hist : hist_vec) {
            hist_weight[id].push_back(cv::compareHist(new_hist, hist, cv::HISTCMP_CORREL)); // Сравниваем и добавляем вес
        }
    }
    return hist_weight; // Возвращаем веса гистограмм
}

// Метод для отрисовки трекеров на изображении
cv::Mat TrackingBox::drawBoxes(cv::Mat& frame, const std::vector<TrackingBox>& boxes) {
    for (const auto& box : boxes) { // Для каждого трекера
        auto [x, y, w, h] = box.shape(); // Получаем форму
        int cx = x + w / 2; // Центральная точка по X
        int cy = y + h / 2; // Центральная точка по Y
        cv::putText(frame, std::to_string(box.id), cv::Point(cx, cy - 7), 0, 1, RedColor, 2); // Выводим ID
        cv::rectangle(frame, cv::Point(x, y), cv::Point(x + w, y + h), GreenColor, 2); // Рисуем прямоугольник
        cv::line(frame, cv::Point(x, cy), cv::Point(x + w, cy), RedColor, 2); // Рисуем горизонтальную линию
    }
    return frame; // Возвращаем изображение с отрисованными трекерами
}

// Новый метод для вычисления IoU между двумя прямоугольниками
double TrackingBox::intersectionOverUnion(const cv::Rect& a, const cv::Rect& b) {
    cv::Rect intersection = a & b; // Пересечение
    return static_cast<double>(intersection.area()) / (a.area() + b.area() - intersection.area()); // Возвращаем IoU
}

// Новый метод для сопоставления трекеров и обнаруженных объектов на основе матрицы IoU
std::tuple<std::vector<std::vector<int>>, std::vector<int>, std::vector<int>> TrackingBox::match(const cv::Mat& IoU, const std::vector<TrackingBox>& trackers, const std::vector<TrackingBox>& detections) {
    std::vector<int> assignment;
    std::vector<std::vector<double>> costMatrix(trackers.size(), std::vector<double>(detections.size()));

    // Заполнение costMatrix на основе IoU
    for (size_t t = 0; t < trackers.size(); ++t) {
        for (size_t d = 0; d < detections.size(); ++d) {
            costMatrix[t][d] = 1 - IoU.at<double>(t, d); // Превращаем в стоимость
        }
    }

    HungarianAlgorithm hungAlgo; // Создаем объект алгоритма Венгерского
    hungAlgo.Solve(costMatrix, assignment); // Решаем задачу сопоставления

    std::vector<std::vector<int>> matches; // Сопоставленные пары
    std::vector<int> unmatched_trackers, unmatched_detections; // Несопоставленные трекеры и обнаружения

    // Определение сопоставлений и несопоставленных трекеров
    for (size_t t = 0; t < trackers.size(); ++t) {
        int matched_det_id = assignment[t];
        bool is_unmatched = matched_det_id == -1 || IoU.at<double>(t, matched_det_id) < 0.3; // Проверка на несопоставленность

        if (is_unmatched) {
            unmatched_trackers.push_back(t); // Если несопоставленный, добавляем в список
        }
        else {
            matches.push_back({ static_cast<int>(t), static_cast<int>(matched_det_id) }); // Сохраняем сопоставление
        }
    }

    // Определение несопоставленных объектов
    for (size_t d = 0; d < detections.size(); ++d) {
        if (std::none_of(matches.begin(), matches.end(), [d](const std::vector<int>& match) { return match[1] == d; })) {
            unmatched_detections.push_back(d); // Если нет сопоставления, добавляем в несопоставленные
        }
    }

    return { matches, unmatched_detections, unmatched_trackers }; // Возвращаем сопоставления и несопоставленные
}
