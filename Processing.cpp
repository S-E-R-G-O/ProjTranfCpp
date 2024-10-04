#include "Processing.h"
#include "TrackingBox.h"

// Реализация конструктора класса Processing
Processing::Processing(const string& fileName1, const string& fileName2) : isFrame(true)
{
    // Открываем видеопотоки: первый и второй
    if (!stream1.open(fileName1) || !stream2.open(fileName2))
    {
        throw runtime_error("Не удалось открыть видеопотоки: " + fileName1 + " или " + fileName2);
    }

    background = Mat(); // Инициализация фона как пустой матрицы
}

// Реализация деструктора
Processing::~Processing()
{
    // Освобождение ресурсов видеопотоков
    stream1.release(); // Освобождение ресурсов первого видеопотока
    stream2.release(); // Освобождение ресурсов второго видеопотока
}

// Метод для обнаружения изменений между двумя видеопотоками
void Processing::detectedChanges()
{
    Mat frame1, frame2, grayFrame1; // Матрицы для хранения кадров и их серой версии

    while (true) // Бесконечный цикл для обработки кадров
    {
        // Чтение кадров из обоих видеопотоков
        stream1 >> frame1;
        stream2 >> frame2;

        // Проверяем, были ли прочитаны оба кадра
        if (frame1.empty() || frame2.empty()) // Проверка на пустые кадры
        {
            cout << "Один из кадров пуст; остановка обработки." << endl;
            break; // Выход из цикла, если хотя бы один кадр пуст
        }

        // Соединяем оба кадра в один
        hconcat(frame1, frame2, frame1);

        // Преобразуем объединенный кадр в оттенки серого
        cvtColor(frame1, grayFrame1, COLOR_BGR2GRAY);

        // Обрабатываем текущий кадр
        processingFrame(frame1, grayFrame1);

        // Проверка нажатия клавиши ESC для выхода из цикла
        if (waitKey(10) == 27)
        {
            break;
        }
    }
}

// Метод для обработки текущего кадра
void Processing::processingFrame(Mat& frame, Mat& grayFrame)
{
    Mat difference, thresh, dilated, frameBlur; // Матрицы для хранения промежуточных результатов

    // Накапливаем фон для обнаружения изменений
    if (background.empty())
    {
        background = Mat::zeros(grayFrame.size(), CV_32F); // Создаем пустую матрицу для фона
        grayFrame.convertTo(background, CV_32F); // Инициализируем фон первым серым кадром
    }
    else
    {
        // Накапливаем фон с заданным весом
        accumulateWeighted(grayFrame, background, 0.15);
    }

    // Вычисляем абсолютную разность между фоном и текущим кадром
    Mat backgroundupd;
    background.convertTo(backgroundupd, CV_8U); // Преобразование фона в 8-битный формат
    absdiff(backgroundupd, grayFrame, difference); // Вычисление разности

    // Применяем пороговую обработку и морфологические операции
    threshold(difference, thresh, 30, 255, THRESH_BINARY); // Пороговая фильтрация
    dilate(thresh, dilated, Mat(), Point(-1, -1), 4); // Увеличение областей с помощью диляции
    GaussianBlur(dilated, frameBlur, Size(5, 5), 0, 0); // Размытие для уменьшения шумов
     
    vector<vector<Point>> contours; // Вектор для хранения найденных контуров
    findContours(frameBlur, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE); // Поиск внешних контуров

    // Создаём блоки отслеживания по найденным контурам
    vector<TrackingBox> boxes = TrackingBox::createBoxes(contours);
    TrackingBox::drawBoxes(frame, boxes); // Рисуем блоки на кадре
    for (const auto& box : boxes) {
        box.processHistogram(frame); // Обработка гистограммы для найденного бокса
    }
    // Отображение результатов
    imshow("Combined Video", frame); // Отображаем объединенное видео
    imshow("Masked Video", thresh); // Отображаем бинарную маску
}
