#include "HungarianMethod.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <limits>

using namespace std;

// ����������� ��� ������ HungarianMethod
HungarianMethod::HungarianMethod(float iou_threshold) : iou_threshold_(iou_threshold) {}

// ������� ��� ���������� ����������� � ����������� (IoU) ���� ���������������
float HungarianMethod::intersectionOverUnion(const cv::Rect& a, const cv::Rect& b) {
    // ���������� ��������� �����������
    int xA = max(a.x, b.x);
    int yA = max(a.y, b.y);
    int xB = min(a.x + a.width, b.x + b.width);
    int yB = min(a.y + a.height, b.y + b.height);

    // ���������� ������� �����������
    int interArea = max(0, xB - xA) * max(0, yB - yA);
    // ������� ��������������� a � b
    float boxAArea = a.width * a.height;
    float boxBArea = b.width * b.height;

    // ������� �������� IoU
    return interArea / (boxAArea + boxBArea - interArea);
}

// ���������� ��������� ����������� ������ ��� ������������� ===========================================
void HungarianMethod::hungarianAlgorithm(const cv::Mat& costMatrix, std::vector<int>& assignment) {
    int n = costMatrix.rows; // ���������� ����� (��������)
    int m = costMatrix.cols; // ���������� �������� (��������)

    // ������� ��� �������� ���������� � ����������� � ������������
    std::vector<int> u(n, 0), v(m, 0), p(m, -1), way(m, -1);
    for (int i = 0; i < n; i++) {
       
        std::vector<int> links(m, -1), mins(m, std::numeric_limits<int>::max());
        std::vector<bool> visited(m, false);
        int j0 = -1, j1 = 0, delta;

        
        for (int j = 0; j < m; j++) {
            mins[j] = costMatrix.at<float>(i, j);
        }

        
        while (true) {
            visited[j1] = true; 
            int i0 = j1, j0_old = j0;
            j0 = -1;
            delta = std::numeric_limits<int>::max();

           
            for (int j = 0; j < m; j++) {
                if (!visited[j]) {
                    int cur = mins[j] - u[i0] - v[j];
                    if (cur < delta) {
                        delta = cur;
                        j1 = j; 
                    }
                }
            }

           
            for (int j = 0; j < m; j++) {
                if (visited[j]) {
                    u[p[j]] += delta; 
                    v[j] -= delta; 
                }
                else {
                    mins[j] -= delta; 
                }
            }

           
            p[j1] = i0;
            if (j0_old == -1) {
                break; 
            }
            else {
                j1 = j0_old; 
            }
        }
    }

    
    assignment.resize(n, -1);
    for (int j = 0; j < m; j++) {
        if (p[j] != -1) {
            assignment[p[j]] = j; // ��������� ������������
        }
    }
}
//====================================================================================

// �������� ������� ��� ������������� �������� � ��������
std::tuple<std::vector<std::vector<int>>, std::vector<int>, std::vector<int>> HungarianMethod::match(const cv::Mat& IoU, const std::vector<cv::Rect>& trackers, const std::vector<cv::Rect>& detections) {
    vector<int> assignment;
    // ���������� ��������� ����������� ���������
    hungarianAlgorithm(-IoU, assignment);
    vector<vector<int>> matches; // ������ ������������
    vector<int> unmatched_trackers, unmatched_detections; // ������ ��������������

    // �������� ������������ �� ������ ������ IoU
    for (size_t t = 0; t < trackers.size(); ++t) {
        // ���� ������������ �� ������� ��� ���� ������ IoU, ��������� � ���������������� 
        if (assignment[t] == -1 || intersectionOverUnion(detections[assignment[t]], trackers[t]) < iou_threshold_) {
            unmatched_trackers.push_back(t);
        }
        else {
            matches.push_back({ static_cast<int>(assignment[t]), static_cast<int>(t) }); // ���������� ���������� ������������
        }
    }

    // ���������, ����� �������� �������� �����������������
    for (size_t d = 0; d < detections.size(); ++d) {
        if (std::find_if(matches.begin(), matches.end(), [d](const std::vector<int>& match) { return match[0] == static_cast<int>(d); }) == matches.end()) {
            unmatched_detections.push_back(d); // ���������� � ����������������
        }
    }

    // ���������� ������������ � ������ ����������������� �������� � ��������
    return { matches, unmatched_detections, unmatched_trackers };
}