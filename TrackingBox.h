#pragma once
#ifndef TRK_BOX
#define TRK_BOX

#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>
#include <string>
#include <unordered_map>

class TrackingBox {
public:
    static cv::Scalar GreenColor;
    static cv::Scalar RedColor;
    static const int det_limArea = 6000;

    TrackingBox(int x, int y, int w, int h);
    std::tuple<int, int, int, int> shape() const;
    cv::Rect rectangle() const; 
    void processHistogram(const cv::Mat& frame) const;
    static std::vector<TrackingBox> createBoxes(const std::vector<std::vector<cv::Point>>& contours);
    static cv::Mat drawBoxes(cv::Mat& frame, const std::vector<TrackingBox>& boxes);
    static std::vector<TrackingBox> trackingCreation(std::vector<TrackingBox>& detections, std::vector<TrackingBox>& trackers);
    //1.Сравнение гистограмм: В процессе...
    //static std::unordered_map<int, std::vector<double>> compareHistograms(const cv::Mat& new_hist);

private:
    int x, y, w, h;
    int id;
    static std::tuple<cv::Mat, cv::Mat, cv::Mat> computeHist(const cv::Mat& region);
    static void printMeanval(const cv::Mat& b_hist, const cv::Mat& g_hist, const cv::Mat& r_hist, const std::string& half);
    static std::unordered_map<int, std::vector<cv::Mat>> track_hist;
    static std::unordered_map<int, std::vector<cv::Mat>> del_hists;
    static int id_counter;

};

#endif // TRK_BOX
