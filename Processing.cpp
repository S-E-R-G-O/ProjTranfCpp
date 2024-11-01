#include "Processing.h"
#include "TrackingBox.h"
#include "IntersertoinOverUnion.h"

// Конструктор класса Processing
Processing::Processing(const string& fileName1, const string& fileName2) : isFrame(true)
{
    // Открываем два потока: один для каждого видео
    if (!stream1.open(fileName1) || !stream2.open(fileName2))
    {
        throw runtime_error("Не удалось открыть файлы: " + fileName1 + " и " + fileName2);
    }

    background = cv::Mat(); // Инициализируем фоновое изображение пустой матрицей
}

// Деструктор класса Processing
Processing::~Processing()
{
    // Освобождаем ресурсы потоков видео
    stream1.release(); // Освобождаем первый поток
    stream2.release(); // Освобождаем второй поток
}

// Метод для обнаружения изменений между кадрами
vector<TrackingBox> Processing::detectedChanges(cv::Mat& frame, cv::Mat& thresh)
{
    cv::Mat frame1, frame2, grayFrame1; // Объявляем матрицы для кадров и их серых версий

    // Читаем текущие кадры из потоков
    stream1 >> frame1;
    stream2 >> frame2;

    // Проверяем, были ли успешно считаны кадры
    if (frame1.empty() || frame2.empty()) // Если один из кадров пустой
    {
        throw runtime_error("Кадры пустые; проверьте входные данные.");
    }

    // Объединяем два кадра в один
    hconcat(frame1, frame2, frame);

    // Конвертируем объединенный кадр в оттенки серого
    cvtColor(frame, grayFrame1, cv::COLOR_BGR2GRAY);

    // Обрабатываем текущий кадр и получаем вектор обнаруженных изменений
    vector<TrackingBox> det = processingFrame(frame, grayFrame1);
    grayFrame1.copyTo(thresh); // Копируем серый кадр

    return det;
}

// Метод для обработки каждого кадра
vector<TrackingBox> Processing::processingFrame(cv::Mat& frame, cv::Mat& grayFrame)
{
    cv::Mat difference, thresh, dilated, frameBlur;

    // Проверяем, инициализировано ли фоновое изображение
    if (background.empty())
    {
        background = cv::Mat::zeros(grayFrame.size(), CV_32F); // Создаем пустое фоновое изображение
        grayFrame.convertTo(background, CV_32F); // Конвертируем текущий серый кадр в  матрицу
    }
    else
    {
        // Обновляем фоновое изображение с помощью сглаживания
        accumulateWeighted(grayFrame, background, 0.15);
    }

    // Обновляемое фоновое изображение конвертируем в 8-битное
    cv::Mat backgroundupd;
    background.convertTo(backgroundupd, CV_8U); // Конвертация фона в 8-битный формат
    absdiff(backgroundupd, grayFrame, difference); // Вычисляем абсолютную разницу между фоном и текущим кадром

    // Применяем пороговое преобразование для выделения изменений
    threshold(difference, thresh, 30, 255, cv::THRESH_BINARY); // Пороговое преобразование
    dilate(thresh, dilated, cv::Mat(), cv::Point(-1, -1), 4); // Дилатация для усиления изменений
    GaussianBlur(dilated, frameBlur, cv::Size(5, 5), 0, 0); // Размытие для уменьшения шумов

    vector<vector<cv::Point>> contours; // Вектор для хранения контуров
    findContours(frameBlur, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE); // Находим контуры на размытой картинке

    // Создаем боксы на основе обнаруженных контуров
    vector<TrackingBox> boxes = TrackingBox::createBoxes(contours);

    // При необходимости можно дополнительно обрабатывать каждый бок
    // for (const auto& box : boxes) {
    //     box.processHistogram(frame); // Метод для обработки гистограммы (не активен)
    // }
    thresh.copyTo(grayFrame); // Копируем пороговое изображение в серый кадр

    return boxes;
}
