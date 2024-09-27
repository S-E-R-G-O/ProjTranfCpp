#include "Processing.h"
#include "TrackingBox.h"

// ���������� ������������ ������ Processing
Processing::Processing(const string& fileName1, const string& fileName2) : isFrame(true)
{
    // ��������� �����������: ������ � ������
    if (!stream1.open(fileName1) || !stream2.open(fileName2))
    {
        throw runtime_error("�� ������� ������� �����������: " + fileName1 + " ��� " + fileName2);
    }

    background = Mat(); // ������������� ���� ��� ������ �������
}

// ���������� �����������
Processing::~Processing()
{
    // ������������ �������� ������������
    stream1.release(); // ������������ �������� ������� �����������
    stream2.release(); // ������������ �������� ������� �����������
}

// ����� ��� ����������� ��������� ����� ����� �������������
void Processing::detectedChanges()
{
    Mat frame1, frame2, grayFrame1; // ������� ��� �������� ������ � �� ����� ������

    while (true) // ����������� ���� ��� ��������� ������
    {
        // ������ ������ �� ����� ������������
        stream1 >> frame1;
        stream2 >> frame2;

        // ���������, ���� �� ��������� ��� �����
        if (frame1.empty() || frame2.empty()) // �������� �� ������ �����
        {
            cout << "���� �� ������ ����; ��������� ���������." << endl;
            break; // ����� �� �����, ���� ���� �� ���� ���� ����
        }

        // ��������� ��� ����� � ����
        hconcat(frame1, frame2, frame1);

        // ����������� ������������ ���� � ������� ������
        cvtColor(frame1, grayFrame1, COLOR_BGR2GRAY);

        // ������������ ������� ����
        processingFrame(frame1, grayFrame1);

        // �������� ������� ������� ESC ��� ������ �� �����
        if (waitKey(10) == 27)
        {
            break;
        }
    }
}

// ����� ��� ��������� �������� �����
void Processing::processingFrame(Mat& frame, Mat& grayFrame)
{
    Mat difference, thresh, dilated, frameBlur; // ������� ��� �������� ������������� �����������

    // ����������� ��� ��� ����������� ���������
    if (background.empty())
    {
        background = Mat::zeros(grayFrame.size(), CV_32F); // ������� ������ ������� ��� ����
        grayFrame.convertTo(background, CV_32F); // �������������� ��� ������ ����� ������
    }
    else
    {
        // ����������� ��� � �������� �����
        accumulateWeighted(grayFrame, background, 0.15);
    }

    // ��������� ���������� �������� ����� ����� � ������� ������
    Mat backgroundupd;
    background.convertTo(backgroundupd, CV_8U); // �������������� ���� � 8-������ ������
    absdiff(backgroundupd, grayFrame, difference); // ���������� ��������

    // ��������� ��������� ��������� � ��������������� ��������
    threshold(difference, thresh, 30, 255, THRESH_BINARY); // ��������� ����������
    dilate(thresh, dilated, Mat(), Point(-1, -1), 4); // ���������� �������� � ������� �������
    GaussianBlur(dilated, frameBlur, Size(5, 5), 0, 0); // �������� ��� ���������� �����

    vector<vector<Point>> contours; // ������ ��� �������� ��������� ��������
    findContours(frameBlur, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE); // ����� ������� ��������

    // ������ ����� ������������ �� ��������� ��������
    vector<TrackingBox> boxes = TrackingBox::createBoxes(contours);
    TrackingBox::drawBoxes(frame, boxes); // ������ ����� �� �����

    // ����������� �����������
    imshow("Combined Video", frame); // ���������� ������������ �����
    imshow("Masked Video", thresh); // ���������� �������� �����
}
