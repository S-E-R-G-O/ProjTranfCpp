#ifndef HUNGARIANMETHOD_H
#define HUNGARIANMETHOD_H
#include <vector>
#include <opencv2/core.hpp>
using namespace std;

class HungarianMethod
{
public:
	HungarianMethod(float iou_threshold = 0.3f);
	tuple<std::vector<std::vector<int>>, std::vector<int>, std::vector<int>> match(const cv::Mat& IoU, const std::vector<cv::Rect>& trackers, const std::vector<cv::Rect>& detections);

private:

	float iou_threshold_;
	float intersectionOverUnion(const cv::Rect& a, const cv::Rect& b);
	void hungarianAlgorithm(const cv::Mat& costMatrix, std::vector<int>& assignment);

};




#endif // !
