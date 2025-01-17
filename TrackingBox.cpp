#include <unordered_map>
#include <iostream>
#include <numeric>
#include "TrackingBox.h" // Заголовочный файл класса отслеживающего бокса
#include "IntersertoinOverUnion.h" // Заголовочный файл для вычисления IoU

// Инициализация статических переменных класса TrackingBox
cv::Scalar TrackingBox::GreenColor = cv::Scalar(100, 255, 0); // Зеленый цвет для отображения
cv::Scalar TrackingBox::RedColor = cv::Scalar(0, 0, 255); // Красный цвет для отображения текста
std::unordered_map<int, std::vector<cv::Mat>> TrackingBox::track_hist; // Хранилище историй отслеживания
std::unordered_map<int, std::vector<cv::Mat>> TrackingBox::del_hists; // Хранилище удаленных историй
int TrackingBox::id_counter = 0; // Счетчик для уникальных идентификаторов

// Конструктор класса TrackingBox
TrackingBox::TrackingBox(int x, int y, int w, int h)
    : x(x), y(y), w(w), h(h), id(id_counter++) {} // Инициализация переменных и уникального ID

// Функция для создания отслеживаемых объектов
std::vector<TrackingBox> TrackingBox::trackingCreation(
    std::vector<TrackingBox>& detections, // Обнаруженные боксы
    std::vector<TrackingBox>& trackers) { // Отслеживаемые боксы

    if (trackers.empty()) {
        return detections; // Если нет отслеживаемых боксов, возвращаем обнаруженные
    }

    if (detections.empty()) {
        // Удаление всех трекеров, если нет обнаружений
        for (auto& t : trackers) {
            if (track_hist.find(t.id) != track_hist.end()) {
                del_hists[t.id] = std::move(track_hist[t.id]);
                track_hist.erase(t.id);
            }
            else {
                del_hists[t.id] = {};
            }
            std::cout << "DEL " << t.id << " " << del_hists[t.id].size() << std::endl;
        }
        trackers.clear();
        return {}; // Возвращаем пустой вектор, если не найдено обнаружение
    }
    //Тесты
    // Отображение информации о обнаружениях и отслеживаемых боксерах
     /*
    std::cout << "\n--------BEGIN--------" << std::endl;
    std::cout << "detection: " << detections.size() << std::endl;
    for (auto& _ : detections) _.Print();

    std::cout << "trackers: " << trackers.size() << std::endl;
    for (auto& _ : trackers) _.Print();
    */
    // Создание матрицы для вычисления IoU
    cv::Mat IoU(trackers.size(), detections.size(), CV_64F);
    for (size_t t = 0; t < trackers.size(); ++t) {
        for (size_t d = 0; d < detections.size(); ++d) {
            IoU.at<double>(t, d) = IntersectOverUnion::intersectionOverUnion(trackers[t].rectangle(), detections[d].rectangle());
        }
    }
    //Тесты
    // Вывод результата матрицы IoU
    /*
    std::cout << "IoU: " << IoU.size() << std::endl;
    std::cout.setf(std::ios::fixed);
    std::cout.precision(2);
    */
    /*
    for (int i = 0; i < IoU.rows; i++) {
        for (int j = 0; j < IoU.cols; j++) {
            std::cout << IoU.at<double>(i, j) << ",\t";
        }
        std::cout << std::endl;
    }
    */
    IntersectOverUnion io; // Создание объекта для метода венгерского
    auto [matches, unmatched_det_ids, unmatched_trk_ids] = io.match(IoU, trackers, detections); // Сопоставление

    // Обработка совпадений
    for (const auto& mtcd : matches) {
        detections[mtcd[1]].id = trackers[mtcd[0]].id; // Присваивание ID от отслеживаемого объекта
        trackers[mtcd[0]] = detections[mtcd[1]]; // Обновление_tracker'а найденным объектом
    }

    // Добавление несопоставленных обнаружений в отслеживаемые боксы
    for (auto det_id : unmatched_det_ids) {
        trackers.push_back(detections[det_id]);
    }

    // Удаление несопоставленных отслеживаемых боксов
    for (auto ut = unmatched_trk_ids.rbegin(); ut != unmatched_trk_ids.rend(); ++ut) {
        if (track_hist.find(trackers[*ut].id) != track_hist.end()) {
            del_hists[trackers[*ut].id] = std::move(track_hist[trackers[*ut].id]); // Сохраняем историю
            track_hist.erase(trackers[*ut].id); // Удаляем из текущего хранилища истории
        }
        else {
            del_hists[trackers[*ut].id] = {}; // Если не было истории, сохраняем пустой вектор
        }
        std::cout << "DEL " << trackers[*ut].id << " " << del_hists[trackers[*ut].id].size() << std::endl;
        trackers.erase(trackers.begin() + *ut); // Удаляем трекер
    }

    return trackers; // Возвращение обновленных отслеживаемых боксов
}

// Возвращает форму бокса
std::tuple<int, int, int, int> TrackingBox::shape() const {
    return std::make_tuple(x, y, w, h);
}

// Возвращает прямоугольник, представляющий TrackingBox
cv::Rect TrackingBox::rectangle() const {
    return cv::Rect(x, y, w, h);
}

// Создает боксы на основе контуров
std::vector<TrackingBox> TrackingBox::createBoxes(const std::vector<std::vector<cv::Point>>& contours) {
    std::vector<TrackingBox> boxes;

    for (const auto& cnt : contours) {
        if (cv::contourArea(cnt) > det_limArea) { // Проверка на минимальную площадь
            cv::Rect rect = cv::boundingRect(cnt); //  Отрисовка прямоугольник вокруг контура
            boxes.emplace_back(rect.x, rect.y, rect.width, rect.height); // Создаем новый TrackingBox
        }
    }
    return boxes; // Возвращаем все созданные боксы
}


// Обработка гистограмм для визуализации цвета в боксе
void TrackingBox::processHistogram(const cv::Mat& frame) {
    auto [x, y, w, h] = this->shape(); // Получаем параметры бокса

    // Проверка выхода за пределы изображения
    if (y + h >= frame.rows || y < 0) {
        return;
    }

    cv::Rect roi_top(x, y, w, h / 2); // Верхняя часть бокса
    cv::Rect roi_bot(x, y + h / 2, w, h / 2); // Нижняя часть бокса

    cv::Mat region_top = frame(roi_top); // Получение верхней области
    cv::Mat region_bot = frame(roi_bot); // Получение нижней области

    // Вычисление гистограммы цветов
    auto [b_hist_top, g_hist_top, r_hist_top] = computeHist(region_top);
    auto [b_hist_bot, g_hist_bot, r_hist_bot] = computeHist(region_bot);

    // Сохранение новой гистограммы
    cv::Mat new_hist_top = (b_hist_top + g_hist_top + r_hist_top) / 3.0; // предполагаемый расчёт новой гистограммы
    cv::Mat new_hist_bot = (b_hist_bot + g_hist_bot + r_hist_bot) / 3.0; // предполагаемый расчёт новой гистограммы

    // Сохраняем новую гистограмму
    if (track_hist.find(this->id) != track_hist.end()) {
        track_hist[this->id].push_back(new_hist_top);
        track_hist[this->id].push_back(new_hist_bot);
    }
    else {
        std::cout << "NEW " << this->id << std::endl;

        // Сравниваем с существующими ID
        auto hist_weight_top = compareHistograms(new_hist_top);
        auto hist_weight_bot = compareHistograms(new_hist_bot);
        // Ищем ID с максимальным значением для верхней гистограммы
        int best_id_top = -1;
        double best_weight_top = -1.0;

        if (!hist_weight_top.empty()) {
            for (const auto& [hw_id, weights] : hist_weight_top) {
                double mean_value = std::accumulate(weights.begin(), weights.end(), 0.0) / weights.size();
                std::cout << "ВЕРХНЯЯ ГИСТОГРАММА: " << hw_id << ": " << mean_value << std::endl;

                // Сравниваем и находим лучший ID
                if (mean_value > best_weight_top) {
                    best_weight_top = mean_value;
                    best_id_top = hw_id;
                }
            }
        }

        // Аналогично для нижней гистограммы
        int best_id_bot = -1;
        double best_weight_bot = -1.0;

        if (!hist_weight_bot.empty()) {
            for (const auto& [hw_id, weights] : hist_weight_bot) {
                double mean_value = std::accumulate(weights.begin(), weights.end(), 0.0) / weights.size();
                std::cout << "НИЖНЯЯ ГИСТОГРАММА: " << hw_id << ": " << mean_value << std::endl;

                // Сравниваем и находим лучший ID
                if (mean_value > best_weight_bot) {
                    best_weight_bot = mean_value;
                    best_id_bot = hw_id;
                }
            }
        }

        // Если найден общий лучший ID, переприсваиваем
        if (best_id_top != -1 && best_id_top == best_id_bot /* || best_id_top > best_id_bot || best_id_bot > best_id_top*/) {
            std::cout << "Обновление ID: " << this->id << " на " << best_id_top << std::endl;
            this->id = best_id_top; // Переприсваиваем ID
        }

        // Выводим информацию о наиболее похожих объектах
        if (best_id_top != -1) {
            std::cout << "Наиболее похожий объект для верхней гистограммы: " << best_id_top << " с весом: " << best_weight_top << std::endl;
        }
        if (best_id_bot != -1) {
            std::cout << "Наиболее похожий объект для нижней гистограммы: " << best_id_bot << " с весом: " << best_weight_bot << std::endl;
        }

        track_hist[this->id].push_back(new_hist_bot);
    }
}

// Вычисление гистограмм для трех каналов цвета
std::tuple<cv::Mat, cv::Mat, cv::Mat> TrackingBox::computeHist(const cv::Mat& region) {
    std::vector<cv::Mat> bgr_planes;
    cv::split(region, bgr_planes); // Разделение на каналы

    cv::Mat histograms[3]; // Массив для хранения гистограмм
    int histSize = 256; // Размер гистограммы
    float range[] = { 0, 256 }; // Диапазон значений
    const float* histRange = { range }; 

    // Вычисление и нормализация гистограмм для каждого канала
    for (int i = 0; i < 3; ++i) {
        if (!bgr_planes[i].empty()) {
            cv::calcHist(&bgr_planes[i], 1, 0, cv::Mat(), histograms[i], 1, &histSize, &histRange, true, false);
            cv::normalize(histograms[i], histograms[i], 0, 1, cv::NORM_MINMAX);
        }
    }

    return { histograms[0], histograms[1], histograms[2] }; // Возвращаем гистограммы
}

// Вывод средних значений для гистограммы
void TrackingBox::printMeanval(const cv::Mat& b_hist, const cv::Mat& g_hist, const cv::Mat& r_hist, const std::string& half) {
    double av_mean = (cv::mean(b_hist)[0] + cv::mean(g_hist)[1] + cv::mean(r_hist)[2]) / 3.0;

    std::cout << "Mean color values in the " << half << " half of the tracking box: ";
    std::cout << "Average: " << av_mean << std::endl; // Вывод значений
}
// Реализация сравнения гистограмм объектов 

std::unordered_map<int, std::vector<double>> TrackingBox::compareHistograms(const cv::Mat& new_hist)
{

    std::unordered_map<int, std::vector<double>> hist_weight;

    if (del_hists.empty())
    {
      return  hist_weight;
    }
    for (const auto& [id, hist_vec] : del_hists)
    {
        for (const auto& hist : hist_vec)
        {
            double correl = cv::compareHist(new_hist, hist, cv::HISTCMP_CORREL);
            hist_weight[id].push_back(correl);
        }
        
    }
    return hist_weight;
}



// Функция для отрисовки боксов на изображении
cv::Mat TrackingBox::drawBoxes(cv::Mat& frame, const std::vector<TrackingBox>& boxes) {
    for (const auto& box : boxes) {
        auto [x, y, w, h] = box.shape(); // Получаем параметры бокса
        int cx = x + w / 2; // Центр по оси X
        int cy = y + h / 2; // Центр по оси Y
        cv::putText(frame, std::to_string(box.id), cv::Point(cx, cy - 7), 0, 1, RedColor, 2); // Вывод ID бокса
        cv::rectangle(frame, cv::Point(x, y), cv::Point(x + w, y + h), GreenColor, 2); // Рисуем рамку бокса
        cv::line(frame, cv::Point(x, cy), cv::Point(x + w, cy), RedColor, 2); // Рисуем горизонтальную линию
    }
    return frame; // Возвращаем обновленное изображение
}
