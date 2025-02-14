#include "TrackingBox.h"
#include "HungarianAlgorithm.h"

// Инициализация статических переменных класса TrackingBox
cv::Scalar TrackingBox::GreenColor = cv::Scalar(100, 255, 0); // Задает зеленый цвет для отрисовки
cv::Scalar TrackingBox::RedColor = cv::Scalar(0, 0, 255); // Задает красный цвет для текста
std::unordered_map<int, std::vector<cv::Mat>> TrackingBox::track_hist; // Хранит гистограммы трекеров
std::unordered_map<int, std::vector<cv::Mat>> TrackingBox::del_hists; // Хранит удаленные гистограммы
int TrackingBox::id_counter = 0; // Счетчик для уникальных ID трекеров
std::vector<int> TrackingBox::updated_ids; // Хранит обновленные ID
std::unordered_map<int, std::vector<int>> similar_ids; // Хранит похожие ID

// Конструктор класса TrackingBox
TrackingBox::TrackingBox(int x, int y, int w, int h)
    : x(x), y(y), w(w), h(h), id(id_counter++) {} // Инициализация координат и размеров, а также уникального ID трекера

// Метод для создания трекеров на основе обнаруженных объектов
std::vector<TrackingBox> TrackingBox::trackingCreation(
    std::vector<TrackingBox>& detections,
    std::vector<TrackingBox>& trackers) {

    // Если нет трекеров, возвращаем обнаруженные
    if (trackers.empty()) return detections;

    // Если нет обнаруженных объектов, удаляем трекеры
    if (detections.empty()) {
        for (auto& t : trackers) {
            del_hists[t.id] = std::move(track_hist[t.id]); // Сохраняем гистограммы в удаленные
            track_hist.erase(t.id); // Удаляем гистограммы
            std::cout << "DEL " << t.id << " " << del_hists[t.id].size() << std::endl; // Выводим информацию об удалении
        }
        trackers.clear(); 
        return {};
    }

    // Создаем матрицу IoU (Intersection over Union) для трекеров и обнаруженных объектов
    cv::Mat IoU(trackers.size(), detections.size(), CV_64F);
    for (size_t t = 0; t < trackers.size(); ++t) {
        for (size_t d = 0; d < detections.size(); ++d) {
            IoU.at<double>(t, d) = TrackingBox::intersectionOverUnion(trackers[t].rectangle(), detections[d].rectangle());
        }
    }

    // Выполняем сопоставление трекеров и обнаруженных объектов
    auto [matches, unmatched_det_ids, unmatched_trk_ids] = match(IoU, trackers, detections);

    // Обновляем ID для сопоставленных объектов
    for (const auto& mtcd : matches) {
        detections[mtcd[1]].id = trackers[mtcd[0]].id;
        trackers[mtcd[0]] = detections[mtcd[1]];
    }

    // Добавляем несопоставленные обнаруженные объекты в трекеры
    for (auto det_id : unmatched_det_ids) {
        trackers.emplace_back(detections[det_id]);
    }

    // Удаляем несопоставленные трекеры
    for (auto ut = unmatched_trk_ids.rbegin(); ut != unmatched_trk_ids.rend(); ++ut) {
        del_hists[trackers[*ut].id] = std::move(track_hist[trackers[*ut].id]);
        track_hist.erase(trackers[*ut].id);
        std::cout << "DEL " << trackers[*ut].id << " " << del_hists[trackers[*ut].id].size() << std::endl;
        trackers.erase(trackers.begin() + *ut);
    }

    return trackers; 
}

// Метод для получения формы трекера
std::tuple<int, int, int, int> TrackingBox::shape() const {
    return { x, y, w, h }; // Возвращает координаты и размеры трекера
}

// Метод для получения прямоугольника трекера
cv::Rect TrackingBox::rectangle() const {
    return cv::Rect(x, y, w, h); // Возвращает прямоугольник, представляющий трекер
}

// Метод для создания трекеров на основе контуров
std::vector<TrackingBox> TrackingBox::createBoxes(const std::vector<std::vector<cv::Point>>& contours) {
    std::vector<TrackingBox> boxes;
    for (const auto& cnt : contours) {
        if (cv::contourArea(cnt) > det_limArea) { // Проверяем минимальную площадь контура
            cv::Rect rect = cv::boundingRect(cnt); // Получаем ограничивающий прямоугольник
            boxes.emplace_back(rect.x, rect.y, rect.width, rect.height); // Создаем новый трекер
        }
    }
    return boxes;
}

// Метод для нахождения лучшего ID на основе гистограмм
std::tuple<int, double> TrackingBox::findBestId(const std::unordered_map<int, std::vector<double>>& hist_weights, const std::string& half) {
    int best_id = -1;
    double best_weight = -1.0;

    for (const auto& [hw_id, weights] : hist_weights) {
        double mean_value = std::accumulate(weights.begin(), weights.end(), 0.0) / weights.size(); // Вычисляем среднее значение весов
        std::cout << half << " ГИСТОГРАММА: " << hw_id << ": " << mean_value << std::endl;

        if (mean_value > best_weight) { // Находим ID с наилучшим весом
            best_weight = mean_value;
            best_id = hw_id;
        }
    }

    return { best_id, best_weight }; // Возвращаем лучший ID и его вес
}

// Метод для обновления ID и вывода информации
void TrackingBox::updateIdAndPrint(int best_id_top, double best_weight_top, int best_id_bot, double best_weight_bot) {
    if (best_id_top != -1) {
        if (best_id_top == best_id_bot) { // Если оба ID совпадают
            std::cout << "Обновление ID: " << this->id << " на " << best_id_top << std::endl;
            this->id = best_id_top; // Обновляем ID
            updated_ids.push_back(best_id_top); // Добавляем в обновленные ID
        }
        std::cout << "Наиболее похожий объект для верхней гистограммы: " << best_id_top << " с весом: " << best_weight_top << std::endl;
        updated_ids.push_back(best_id_top); // Добавляем ID верхней гистограммы
        if (this->id != best_id_top) {
            similar_ids[this->id].push_back(best_id_top); // Сохраняем похожий ID
        }
    }

    if (best_id_bot != -1) {
        std::cout << "Наиболее похожий объект для нижней гистограммы: " << best_id_bot << " с весом: " << best_weight_bot << std::endl;
        updated_ids.push_back(best_id_bot); // Добавляем ID нижней гистограммы
        if (this->id != best_id_bot) {
            similar_ids[this->id].push_back(best_id_bot); // Сохраняем похожий ID
        }
    }

    propagateIds(); // Распределяем ID
}

// Метод для распространения ID
void TrackingBox::propagateIds() {
    std::unordered_map<int, int> final_ids;

    for (const auto& [id, similar] : similar_ids) {
        int current_id = id;
        for (int sim_id : similar) { // Находим финальный ID для каждого похожего ID
            if (final_ids.count(sim_id)) {
                current_id = final_ids[sim_id]; // Если похожий ID уже имеет финальный, обновляем current_id
            }
        }
        final_ids[id] = current_id; // Сохраняем финальный ID
    }

    if (final_ids.count(this->id)) { // Если наш текущий ID имеет финальный
        this->id = final_ids[this->id];
        std::cout << "Обновленный ID после распространения: " << this->id << std::endl;
    }
}

// Метод для обработки гистограммы
void TrackingBox::processHistogram(const cv::Mat& frame) {
    auto [x, y, w, h] = this->shape(); // Получаем координаты и размеры трекера

    if (y + h >= frame.rows || y < 0) { // Проверяем, находится ли трекер в пределах изображения
        return;
    }

    // Разделяем регион на верхнюю и нижнюю части
    cv::Mat region_top = frame(cv::Rect(x, y, w, h / 2));
    cv::Mat region_bot = frame(cv::Rect(x, y + h / 2, w, h / 2));

    // Вычисляем гистограммы для верхней и нижней частей
    auto [b_hist_top, g_hist_top, r_hist_top] = computeHist(region_top);
    auto [b_hist_bot, g_hist_bot, r_hist_bot] = computeHist(region_bot);

    // Объединяем гистограммы
    auto new_hist_top = (b_hist_top + g_hist_top + r_hist_top) / 3.0;
    auto new_hist_bot = (b_hist_bot + g_hist_bot + r_hist_bot) / 3.0;

    // Обработка новых гистограмм
    if (track_hist.count(this->id)) {
        track_hist[this->id].emplace_back(new_hist_top); // Добавляем верхнюю гистограмму
        track_hist[this->id].emplace_back(new_hist_bot); // Добавляем нижнюю гистограмму
    }
    else {
        std::cout << "NEW " << this->id << std::endl; // Если ID новый, выводим сообщение

        auto hist_weight_top = compareHistograms(new_hist_top); // Сравниваем верхнюю гистограмму
        auto hist_weight_bot = compareHistograms(new_hist_bot); // Сравниваем нижнюю гистограмму

        auto [best_id_top, best_weight_top] = findBestId(hist_weight_top, "ВЕРХНЯЯ"); // Находим лучший ID для верхней
        auto [best_id_bot, best_weight_bot] = findBestId(hist_weight_bot, "НИЖНЯЯ"); // Находим лучший ID для нижней

        updateIdAndPrint(best_id_top, best_weight_top, best_id_bot, best_weight_bot); // Обновляем ID
        track_hist[this->id].emplace_back(new_hist_bot); // Добавляем нижнюю гистограмму
    }
}

// Метод для вычисления гистограммы
std::tuple<cv::Mat, cv::Mat, cv::Mat> TrackingBox::computeHist(const cv::Mat& region) {
    std::vector<cv::Mat> bgr_planes;
    cv::split(region, bgr_planes); // Разделяем регион на каналы BGR

    cv::Mat histograms[3];
    const int histSize = 256; // Размер гистограммы
    const float range[] = { 0, 256 }; // Диапазон значений
    const float* histRange = { range };

    for (int i = 0; i < 3; ++i) {
        if (!bgr_planes[i].empty()) {
            cv::calcHist(&bgr_planes[i], 1, 0, cv::Mat(), histograms[i], 1, &histSize, &histRange, true, false);
            cv::normalize(histograms[i], histograms[i], 0, 1, cv::NORM_MINMAX); // Нормализуем гистограммы
        }
    }

    return { histograms[0], histograms[1], histograms[2] }; // Возвращаем гистограммы для каждого канала
}

// Метод для сравнения гистограмм
std::unordered_map<int, std::vector<double>> TrackingBox::compareHistograms(const cv::Mat& new_hist) {
    std::unordered_map<int, std::vector<double>> hist_weight;

    for (const auto& [id, hist_vec] : del_hists) { // Проходим по всем удаленным гистограммам
        for (const auto& hist : hist_vec) {
            hist_weight[id].push_back(cv::compareHist(new_hist, hist, cv::HISTCMP_CORREL)); // Сравниваем гистограммы
        }
    }
    return hist_weight; // Возвращаем веса гистограмм
}

// Метод для отрисовки трекеров на изображении
cv::Mat TrackingBox::drawBoxes(cv::Mat& frame, const std::vector<TrackingBox>& boxes) {
    for (const auto& box : boxes) {
        int displayId = box.getId(); // Получаем ID текущего объекта

        // Проверяем, есть ли гистограмма для данного ID
        if (TrackingBox::track_hist.count(displayId) > 0 && !TrackingBox::track_hist[displayId].empty()) {
            auto [x, y, w, h] = box.shape();
            int cx = x + w / 2; // Центр по X
            int cy = y + h / 2; // Центр по Y

            // Отрисовываем текст и рамку только если гистограмма существует
            cv::putText(frame, std::to_string(displayId), cv::Point(cx, cy - 7), 0, 1, RedColor, 2); // Рисуем ID
            cv::rectangle(frame, cv::Point(x, y), cv::Point(x + w, y + h), GreenColor, 2); // Рисуем рамку
            cv::line(frame, cv::Point(x, cy), cv::Point(x + w, cy), RedColor, 2); // Рисуем линию
        }
    }
    return frame; // Возвращаем обновленное изображение
}

// Новый метод для вычисления IoU между двумя прямоугольниками
double TrackingBox::intersectionOverUnion(const cv::Rect& a, const cv::Rect& b) {
    cv::Rect intersection = a & b; // Находим пересечение
    return static_cast<double>(intersection.area()) / (a.area() + b.area() - intersection.area()); // Вычисляем IoU
}

// Новый метод для сопоставления трекеров и обнаруженных объектов на основе матрицы IoU
std::tuple<std::vector<std::vector<int>>, std::vector<int>, std::vector<int>> TrackingBox::match(const cv::Mat& IoU, const std::vector<TrackingBox>& trackers, const std::vector<TrackingBox>& detections) {
    std::vector<int> assignment; // Список сопоставлений
    std::vector<std::vector<double>> costMatrix(trackers.size(), std::vector<double>(detections.size())); // Матрица затрат

    // Заполнение costMatrix на основе IoU
    for (size_t t = 0; t < trackers.size(); ++t) {
        for (size_t d = 0; d < detections.size(); ++d) {
            costMatrix[t][d] = 1 - IoU.at<double>(t, d); 
        }
    }

    HungarianAlgorithm hungAlgo; // Создаем объект Венгерского алгоритма 
    hungAlgo.Solve(costMatrix, assignment); // Решаем задачу сопоставления

    std::vector<std::vector<int>> matches; // Список сопоставлений
    std::vector<int> unmatched_trackers, unmatched_detections; // Списки несопоставленных трекеров и объектов

    // Определение сопоставлений и несопоставленных трекеров
    for (size_t t = 0; t < trackers.size(); ++t) {
        int matched_det_id = assignment[t];
        if (matched_det_id == -1 || IoU.at<double>(t, matched_det_id) < 0.3) { // Проверяем порог IoU
            unmatched_trackers.push_back(t); // Если трекер не сопоставлен, добавляем его в список несопоставленных
        }
        else {
            matches.push_back({ static_cast<int>(t), static_cast<int>(matched_det_id) }); // Сохраняем сопоставленные пары
        }
    }

    // Определение несопоставленных объектов
    for (size_t d = 0; d < detections.size(); ++d) {
        if (std::none_of(matches.begin(), matches.end(), [d](const std::vector<int>& match) { return match[1] == d; })) { // Проверяем на сопоставление
            unmatched_detections.push_back(d); // Если объект не сопоставлен, добавляем его в список
        }
    }

    return { matches, unmatched_detections, unmatched_trackers }; // Возвращаем сопоставления и несопоставленные
}
