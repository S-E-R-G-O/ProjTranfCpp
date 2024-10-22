#include <unordered_map>
#include <iostream>
#include "TrackingBox.h"
#include "HungarianMethod.h"

// Static member definitions
cv::Scalar TrackingBox::GreenColor = cv::Scalar(100, 255, 0);
cv::Scalar TrackingBox::RedColor = cv::Scalar(0, 0, 255);
std::unordered_map<int, std::vector<cv::Mat>> TrackingBox::track_hist;
std::unordered_map<int, std::vector<cv::Mat>> TrackingBox::del_hists;
int TrackingBox::id_counter = 0;

// TrackingBox constructor
TrackingBox::TrackingBox(int x, int y, int w, int h)
    : x(x), y(y), w(w), h(h), id(id_counter++) { }

// Implement the new trackingCreation method
std::vector<TrackingBox> TrackingBox::trackingCreation(
    std::vector<TrackingBox>& detections,
    std::vector<TrackingBox>& trackers) {

    if (trackers.empty()) {
        return detections;
    }

    if (detections.empty()) {
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
        return {};
    }

    // Create the IoU matrix
    cv::Mat IoU(detections.size(), trackers.size(), CV_32F);
    for (size_t i = 0; i < detections.size(); ++i) {
        for (size_t j = 0; j < trackers.size(); ++j) {
            IoU.at<float>(i, j) = HungarianMethod::intersectionOverUnion(detections[i].rectangle(), trackers[j].rectangle());
        }
    }

    // Perform the matching
    HungarianMethod hm;
    auto [matches, unmatched_det_ids, unmatched_trk_ids] = hm.match(IoU, trackers, detections);

    // Update detections and trackers accordingly
    for (const auto& mtcd : matches) {
        detections[mtcd[0]].id = trackers[mtcd[1]].id;
        trackers[mtcd[1]] = detections[mtcd[0]];
    }

    for (auto det_id : unmatched_det_ids) {
        trackers.push_back(detections[det_id]);
    }

    // Remove unmatched trackers from the end to prevent index issues
    for (auto ut = unmatched_trk_ids.rbegin(); ut != unmatched_trk_ids.rend(); ++ut) {
        if (track_hist.find(trackers[*ut].id) != track_hist.end()) {
            del_hists[trackers[*ut].id] = std::move(track_hist[trackers[*ut].id]);
            track_hist.erase(trackers[*ut].id);
        }
        std::cout << "DEL " << trackers[*ut].id << " " << del_hists[trackers[*ut].id].size() << std::endl;
        trackers.erase(trackers.begin() + *ut);
    }

    return trackers;
}

std::tuple<int, int, int, int> TrackingBox::shape() const {
    return std::make_tuple(x, y, w, h);
}

cv::Rect TrackingBox::rectangle() const {
    return cv::Rect(x, y, w, h); // Return cv::Rect directly
}

std::vector<TrackingBox> TrackingBox::createBoxes(const std::vector<std::vector<cv::Point>>& contours) {
    std::vector<TrackingBox> boxes;

    for (const auto& cnt : contours) {
        if (cv::contourArea(cnt) > det_limArea) {
            cv::Rect rect = cv::boundingRect(cnt);
            boxes.emplace_back(rect.x, rect.y, rect.width, rect.height);
        }
    }
    return boxes;
}

void TrackingBox::processHistogram(const cv::Mat& frame) const {
    auto [x, y, w, h] = this->shape();

    if (y + h >= frame.rows || y < 0) {
        return;
    }

    cv::Rect roi_top(x, y, w, h / 2);
    cv::Rect roi_bot(x, y + h / 2, w, h / 2);

    cv::Mat region_top = frame(roi_top);
    cv::Mat region_bot = frame(roi_bot);

    auto [b_hist_top, g_hist_top, r_hist_top] = computeHist(region_top);
    auto [b_hist_bot, g_hist_bot, r_hist_bot] = computeHist(region_bot);

    printMeanval(b_hist_top, g_hist_top, r_hist_top, "upper");
    printMeanval(b_hist_bot, g_hist_bot, r_hist_bot, "lower");
}

std::tuple<cv::Mat, cv::Mat, cv::Mat> TrackingBox::computeHist(const cv::Mat& region) {
    std::vector<cv::Mat> bgr_planes;
    cv::split(region, bgr_planes);

    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };

    cv::Mat b_hist, g_hist, r_hist;

    cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, true, false);
    cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, true, false);
    cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, true, false);

    cv::normalize(b_hist, b_hist, 0, 1, cv::NORM_MINMAX);
    cv::normalize(g_hist, g_hist, 0, 1, cv::NORM_MINMAX);
    cv::normalize(r_hist, r_hist, 0, 1, cv::NORM_MINMAX);

    return { b_hist, g_hist, r_hist };
}



void TrackingBox::printMeanval(const cv::Mat& b_hist, const cv::Mat& g_hist, const cv::Mat& r_hist, const std::string& half) {
    double b_mean = cv::mean(b_hist)[0];
    double g_mean = cv::mean(g_hist)[0];
    double r_mean = cv::mean(r_hist)[0];

    std::cout << "Mean color values in the " << half << " half of the tracking box: ";
    std::cout << "Blue: " << b_mean << ", Green: " << g_mean << ", Red: " << r_mean << std::endl;
}

cv::Mat TrackingBox::drawBoxes(cv::Mat& frame, const std::vector<TrackingBox>& boxes) {
    for (const auto& box : boxes) {
        auto [x, y, w, h] = box.shape();
        int cx = x + w / 2;
        int cy = y + h / 2;

        cv::rectangle(frame, cv::Point(x, y), cv::Point(x + w, y + h), GreenColor, 2);
        cv::line(frame, cv::Point(x, cy), cv::Point(x + w, cy), RedColor, 2);
    }
    return frame;
}
