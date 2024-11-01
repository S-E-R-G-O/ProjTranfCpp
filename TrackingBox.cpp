#include <unordered_map>
#include <iostream>
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
        /*
        Удаление всех_tracker'ов, если нет обнаружений (закомментировано)
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
        */
        return {}; // Возвращаем пустой вектор, если не найдено обнаружение
    }

    // Отображение информации о обнаружениях и отслеживаемых боксер
    std::cout << "\n--------BEGIN--------" << std::endl;
    std::cout << "detection: " << detections.size() << std::endl;
    for (auto& _ : detections) _.Print();

    std::cout << "trackers: " << trackers.size() << std::endl;
    for (auto& _ : trackers) _.Print();

    // Создание матрицы для вычисления IoU
    cv::Mat IoU(trackers.size(), detections.size(), CV_64F);
    for (size_t t = 0; t < trackers.size(); ++t) {
        for (size_t d = 0; d < detections.size(); ++d) {
            IoU.at<double>(t, d) = IntersectOverUnion::intersectionOverUnion(trackers[t].rectangle(), detections[d].rectangle());
        }
    }

    // Вывод результата матрицы IoU
    std::cout << "IoU: " << IoU.size() << std::endl;
    std::cout.setf(std::ios::fixed);
    std::cout.precision(2);

    for (int i = 0; i < IoU.rows; i++) {
        for (int j = 0; j < IoU.cols; j++) {
            std::cout << IoU.at<double>(i, j) << ",\t";
        }
        std::cout << std::endl;
    }

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
            del_hists[trackers[*ut].id] = std::move(track_hist[trackers[*ut].id]);
            track_hist.erase(trackers[*ut].id);
        }
        std::cout << "DEL " << trackers[*ut].id << " " << del_hists[trackers[*ut].id].size() << std::endl;
        trackers.erase(trackers.begin() + *ut);
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
void TrackingBox::processHistogram(const cv::Mat& frame) const {
    auto [x, y, w, h] = this->shape(); // Получаем положение и размеры бокса

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

    // Вывод средних значений гистограммы
    printMeanval(b_hist_top, g_hist_top, r_hist_top, "upper");
    printMeanval(b_hist_bot, g_hist_bot, r_hist_bot, "lower");
}

// Вычисление гистограмм для трех каналов цвета
std::tuple<cv::Mat, cv::Mat, cv::Mat> TrackingBox::computeHist(const cv::Mat& region) {
    std::vector<cv::Mat> bgr_planes; // Вектор для цветовых каналов
    cv::split(region, bgr_planes); // Разделение изображения на каналы

    int histSize = 256; // Размер гистограммы
    float range[] = { 0, 256 }; // Диапазон значений
    const float* histRange = { range }; // Указатель на диапазон

    cv::Mat b_hist, g_hist, r_hist; // Гистограммы для каждого канала

    // Вычисление гистограммы для каждого канала
    cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, true, false);
    cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, true, false);
    cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, true, false);

    // Нормализация гистограмм
    cv::normalize(b_hist, b_hist, 0, 1, cv::NORM_MINMAX);
    cv::normalize(g_hist, g_hist, 0, 1, cv::NORM_MINMAX);
    cv::normalize(r_hist, r_hist, 0, 1, cv::NORM_MINMAX);

    return { b_hist, g_hist, r_hist }; // Возвращаем гистограммы
}

// Вывод средних значений для гистограммы
void TrackingBox::printMeanval(const cv::Mat& b_hist, const cv::Mat& g_hist, const cv::Mat& r_hist, const std::string& half) {
    double b_mean = cv::mean(b_hist)[0]; // Среднее значение синего канала
    double g_mean = cv::mean(g_hist)[0]; // Среднее значение зеленого канала
    double r_mean = cv::mean(r_hist)[0]; // Среднее значение красного канала

    std::cout << "Mean color values in the " << half << " half of the tracking box: ";
    std::cout << "Blue: " << b_mean << ", Green: " << g_mean << ", Red: " << r_mean << std::endl; // Вывод значений
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
