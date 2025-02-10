#include "TrackingBox.h"
#include "HungarianAlgorithm.h"

// Инициализация статических переменных класса TrackingBox
cv::Scalar TrackingBox::GreenColor = cv::Scalar(100, 255, 0);
cv::Scalar TrackingBox::RedColor = cv::Scalar(0, 0, 255);
std::unordered_map<int, std::vector<cv::Mat>> TrackingBox::track_hist;
std::unordered_map<int, std::vector<cv::Mat>> TrackingBox::del_hists;
int TrackingBox::id_counter = 0;
std::vector<int> TrackingBox::updated_ids;
std::unordered_map<int, std::vector<int>> similar_ids;

// Конструктор класса TrackingBox
TrackingBox::TrackingBox(int x, int y, int w, int h)
    : x(x), y(y), w(w), h(h), id(id_counter++) {}

// Метод для создания трекеров на основе обнаруженных объектов
std::vector<TrackingBox> TrackingBox::trackingCreation(
    std::vector<TrackingBox>& detections,
    std::vector<TrackingBox>& trackers) {

    if (trackers.empty()) return detections;
    if (detections.empty()) {
        for (auto& t : trackers) {
            del_hists[t.id] = std::move(track_hist[t.id]);
            track_hist.erase(t.id);
            std::cout << "DEL " << t.id << " " << del_hists[t.id].size() << std::endl;
        }
        trackers.clear();
        return {};
    }

    cv::Mat IoU(trackers.size(), detections.size(), CV_64F);
    for (size_t t = 0; t < trackers.size(); ++t) {
        for (size_t d = 0; d < detections.size(); ++d) {
            IoU.at<double>(t, d) = TrackingBox::intersectionOverUnion(trackers[t].rectangle(), detections[d].rectangle());
        }
    }

    auto [matches, unmatched_det_ids, unmatched_trk_ids] = match(IoU, trackers, detections);

    for (const auto& mtcd : matches) {
        detections[mtcd[1]].id = trackers[mtcd[0]].id;
        trackers[mtcd[0]] = detections[mtcd[1]];
    }

    for (auto det_id : unmatched_det_ids) {
        trackers.emplace_back(detections[det_id]);
    }

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
    return { x, y, w, h };
}

// Метод для получения прямоугольника трекера
cv::Rect TrackingBox::rectangle() const {
    return cv::Rect(x, y, w, h);
}

// Метод для создания трекеров на основе контуров
std::vector<TrackingBox> TrackingBox::createBoxes(const std::vector<std::vector<cv::Point>>& contours) {
    std::vector<TrackingBox> boxes;
    for (const auto& cnt : contours) {
        if (cv::contourArea(cnt) > det_limArea) {
            auto rect = cv::boundingRect(cnt);
            boxes.emplace_back(rect.x, rect.y, rect.width, rect.height);
        }
    }
    return boxes;
}

// Метод для нахождения лучшего ID на основе гистограмм
std::tuple<int, double> TrackingBox::findBestId(const std::unordered_map<int, std::vector<double>>& hist_weights, const std::string& half) {
    int best_id = -1;
    double best_weight = -1.0;

    for (const auto& [hw_id, weights] : hist_weights) {
        double mean_value = std::accumulate(weights.begin(), weights.end(), 0.0) / weights.size();
        std::cout << half << " ГИСТОГРАММА: " << hw_id << ": " << mean_value << std::endl;

        if (mean_value > best_weight) {
            best_weight = mean_value;
            best_id = hw_id;
        }
    }

    return { best_id, best_weight };
}

// Метод для обновления ID и вывода информации
void TrackingBox::updateIdAndPrint(int best_id_top, double best_weight_top, int best_id_bot, double best_weight_bot) {
    if (best_id_top != -1) {
        if (best_id_top == best_id_bot) {
            std::cout << "Обновление ID: " << this->id << " на " << best_id_top << std::endl;
            this->id = best_id_top;
            updated_ids.push_back(best_id_top);
        }
        std::cout << "Наиболее похожий объект для верхней гистограммы: " << best_id_top << " с весом: " << best_weight_top << std::endl;
        updated_ids.push_back(best_id_top);
        if (this->id != best_id_top) {
            similar_ids[this->id].push_back(best_id_top);
        }
    }

    if (best_id_bot != -1) {
        std::cout << "Наиболее похожий объект для нижней гистограммы: " << best_id_bot << " с весом: " << best_weight_bot << std::endl;
        updated_ids.push_back(best_id_bot);
        if (this->id != best_id_bot) {
            similar_ids[this->id].push_back(best_id_bot);
        }
    }

    propagateIds();
}

// Метод для распространения ID
void TrackingBox::propagateIds() {
    std::unordered_map<int, int> final_ids;

    for (const auto& [id, similar] : similar_ids) {
        int current_id = id;
        for (int sim_id : similar) {
            if (final_ids.count(sim_id)) {
                current_id = final_ids[sim_id];
            }
        }
        final_ids[id] = current_id;
    }

    if (final_ids.count(this->id)) {
        this->id = final_ids[this->id];
        std::cout << "Обновленный ID после распространения: " << this->id << std::endl;
    }
}

// Метод для обработки гистограммы
void TrackingBox::processHistogram(const cv::Mat& frame) {
    auto [x, y, w, h] = this->shape();

    if (y + h >= frame.rows || y < 0) {
        return;
    }

    // Разделение региона на верхнюю и нижнюю части
    cv::Mat region_top = frame(cv::Rect(x, y, w, h / 2));
    cv::Mat region_bot = frame(cv::Rect(x, y + h / 2, w, h / 2));

    auto [b_hist_top, g_hist_top, r_hist_top] = computeHist(region_top);
    auto [b_hist_bot, g_hist_bot, r_hist_bot] = computeHist(region_bot);

    auto new_hist_top = (b_hist_top + g_hist_top + r_hist_top) / 3.0;
    auto new_hist_bot = (b_hist_bot + g_hist_bot + r_hist_bot) / 3.0;

    // Обработка новых гистограмм
    if (track_hist.count(this->id)) {
        track_hist[this->id].emplace_back(new_hist_top);
        track_hist[this->id].emplace_back(new_hist_bot);
    }
    else {
        std::cout << "NEW " << this->id << std::endl;

        auto hist_weight_top = compareHistograms(new_hist_top);
        auto hist_weight_bot = compareHistograms(new_hist_bot);

        auto [best_id_top, best_weight_top] = findBestId(hist_weight_top, "ВЕРХНЯЯ");
        auto [best_id_bot, best_weight_bot] = findBestId(hist_weight_bot, "НИЖНЯЯ");

        updateIdAndPrint(best_id_top, best_weight_top, best_id_bot, best_weight_bot);
        track_hist[this->id].emplace_back(new_hist_bot);
    }
}

// Метод для вычисления гистограммы
std::tuple<cv::Mat, cv::Mat, cv::Mat> TrackingBox::computeHist(const cv::Mat& region) {
    std::vector<cv::Mat> bgr_planes;
    cv::split(region, bgr_planes);

    cv::Mat histograms[3];
    const int histSize = 256;
    const float range[] = { 0, 256 };
    const float* histRange = { range };

    for (int i = 0; i < 3; ++i) {
        if (!bgr_planes[i].empty()) {
            cv::calcHist(&bgr_planes[i], 1, 0, cv::Mat(), histograms[i], 1, &histSize, &histRange, true, false);
            cv::normalize(histograms[i], histograms[i], 0, 1, cv::NORM_MINMAX);
        }
    }

    return { histograms[0], histograms[1], histograms[2] };
}

// Метод для сравнения гистограмм
std::unordered_map<int, std::vector<double>> TrackingBox::compareHistograms(const cv::Mat& new_hist) {
    std::unordered_map<int, std::vector<double>> hist_weight;

    for (const auto& [id, hist_vec] : del_hists) {
        for (const auto& hist : hist_vec) {
            hist_weight[id].push_back(cv::compareHist(new_hist, hist, cv::HISTCMP_CORREL));
        }
    }
    return hist_weight;
}

// Метод для отрисовки трекеров на изображении
cv::Mat TrackingBox::drawBoxes(cv::Mat& frame, const std::vector<TrackingBox>& boxes) {
    for (const auto& box : boxes) {
        auto [x, y, w, h] = box.shape();
        int cx = x + w / 2;
        int cy = y + h / 2;

        // Используем обновленный ID для отрисовки
        int displayId = box.getId(); // Получаем текущий ID, который мог быть обновлен
        cv::putText(frame, std::to_string(displayId), cv::Point(cx, cy - 7), 0, 1, RedColor, 2);
        cv::rectangle(frame, cv::Point(x, y), cv::Point(x + w, y + h), GreenColor, 2);
        cv::line(frame, cv::Point(x, cy), cv::Point(x + w, cy), RedColor, 2);
    }
    return frame;
}

// Новый метод для вычисления IoU между двумя прямоугольниками
double TrackingBox::intersectionOverUnion(const cv::Rect& a, const cv::Rect& b) {
    cv::Rect intersection = a & b;
    return static_cast<double>(intersection.area()) / (a.area() + b.area() - intersection.area());
}

// Новый метод для сопоставления трекеров и обнаруженных объектов на основе матрицы IoU
std::tuple<std::vector<std::vector<int>>, std::vector<int>, std::vector<int>> TrackingBox::match(const cv::Mat& IoU, const std::vector<TrackingBox>& trackers, const std::vector<TrackingBox>& detections) {
    std::vector<int> assignment;
    std::vector<std::vector<double>> costMatrix(trackers.size(), std::vector<double>(detections.size()));

    // Заполнение costMatrix на основе IoU
    for (size_t t = 0; t < trackers.size(); ++t) {
        for (size_t d = 0; d < detections.size(); ++d) {
            costMatrix[t][d] = 1 - IoU.at<double>(t, d);
        }
    }

    HungarianAlgorithm hungAlgo;
    hungAlgo.Solve(costMatrix, assignment);

    std::vector<std::vector<int>> matches;
    std::vector<int> unmatched_trackers, unmatched_detections;

    // Определение сопоставлений и несопоставленных трекеров
    for (size_t t = 0; t < trackers.size(); ++t) {
        int matched_det_id = assignment[t];
        if (matched_det_id == -1 || IoU.at<double>(t, matched_det_id) < 0.3) {
            unmatched_trackers.push_back(t);
        }
        else {
            matches.push_back({ static_cast<int>(t), static_cast<int>(matched_det_id) });
        }
    }

    // Определение несопоставленных объектов
    for (size_t d = 0; d < detections.size(); ++d) {
        if (std::none_of(matches.begin(), matches.end(), [d](const std::vector<int>& match) { return match[1] == d; })) {
            unmatched_detections.push_back(d);
        }
    }

    return { matches, unmatched_detections, unmatched_trackers };
}
