#include "TrackingBox.h"

// ����������� ����������� ����������
cv::Scalar TrackingBox::GreenColor = cv::Scalar(100, 255, 0); // ������� ����
cv::Scalar TrackingBox::RedColor = cv::Scalar(0, 0, 255); // ������� ����

// �����������, ����������� ���������� (x, y) � ������� (w, h) ��������������
TrackingBox::TrackingBox(int x, int y, int w, int h) : x(x), y(y), w(w), h(h) {}

// ����� ��� ��������� ����� (���������� � �������) �������������� � ���� �������
std::tuple<int, int, int, int> TrackingBox::shape() const {
    return std::make_tuple(x, y, w, h);
}

// ����� ��� ��������� ��������� �������������� � ���� �������
std::vector<int> TrackingBox::rectangle() const {
    return { x, y, x + w, y + h }; // ���������� ������, ���������� ���������� ������ �������� � ������� ������� �����
}

// ����� ��� �������� ������� TrackingBox �� ��������
std::vector<TrackingBox> TrackingBox::createBoxes(const std::vector<std::vector<cv::Point>>& contours) {
    std::vector<TrackingBox> boxes; // ������ ��� �������� ��������� ������

    for (const auto& cnt : contours) {
        // �������� ������� ������� �� ����������� ��������
        if (cv::contourArea(cnt) > det_limArea) {
            cv::Rect rect = cv::boundingRect(cnt); // �������� �������������� ������������� ��� �������
            // ��������� ����� TrackingBox �� ������ ��������������� ��������������
            boxes.emplace_back(rect.x, rect.y, rect.width, rect.height);
        }
    }
    return boxes; // ���������� ������ ��������� ������
}

// ����� ��� �������� ����������� �� �������� ���������������


// ����� ��� ��������� ������ �� �����������
cv::Mat TrackingBox::drawBoxes(cv::Mat& frame, const std::vector<TrackingBox>& boxes) {
    for (const auto& box : boxes) {
        auto [x, y, w, h] = box.shape(); // �������� ���������� � ������� �����
        int cx = x + w / 2; // ������� ����� ����� �� ��� X
        int cy = y + h / 2; // ������� ����� ����� �� ��� Y

        // ������ ������������� �� �����������
        cv::rectangle(frame, cv::Point(x, y), cv::Point(x + w, y + h), GreenColor, 2);
        // ������ �������������� ����� � ������ �����
        cv::line(frame, cv::Point(x, cy), cv::Point(x + w, cy), RedColor, 2);
    }
    return frame; // ���������� ���������������� �����������
}
