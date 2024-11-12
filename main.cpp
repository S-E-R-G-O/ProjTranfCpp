#include "Processing.h"
#include "TrackingBox.h"
#include <iostream>



#include "Processing.h"
#include "TrackingBox.h"
#include <iostream>



int main() {
    cv::Mat frame, threshold;
    std::vector<TrackingBox> detection, tracking;
    setlocale(LC_ALL, "Russian");
    try {
        Processing processing("C:/Users/Main/Desktop/123/video1.avi", "C:/Users/Main/Desktop/123/video2.avi");

        while (true)
        {
            detection = processing.detectedChanges(frame, threshold);
            tracking = TrackingBox::trackingCreation(detection, tracking);
            //строим гистограмы для отслеживаемых объектов
            for (auto& box : tracking) {
                box.processHistogram(frame); // это будет работать, так как box - не const
            }

            TrackingBox::drawBoxes(frame, detection);

            if (cv::waitKey(10) == 27)
            {
                break;
            }

            imshow("Combined Video", frame);
            imshow("Masked Video", threshold);
        }
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
}
