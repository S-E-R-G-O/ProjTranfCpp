#define BOOST_TEST_MODULE mytests
#include <boost/test/included/unit_test.hpp>
#include <iostream> 
#include <opencv2/opencv.hpp> 

// Включаем необходимые файлы для тестирования
#include "..\transfCpp\Processing.cpp"
#include "..\transfCpp\IntersertoinOverUnion.cpp"
#include "..\transfCpp\HungarianAlgorithm.cpp"
#include "..\transfCpp\TrackingBox.cpp"

// Вспомогательная функция для создания тестового видео кадра
cv::Mat createTestFrame(int value) {
    cv::Mat frame(720, 5120, CV_8UC3, cv::Scalar(value, value, value)); // Создаем серый кадр
    return frame;
}
// Тесты для класса Processing ===================================================================================
BOOST_AUTO_TEST_SUITE(ProcessingTests)
// Тест для проверки, что класс Processing может быть сконструирован без исключений
BOOST_AUTO_TEST_CASE(constructor_test) {
    BOOST_CHECK_NO_THROW(Processing processing("C:/Users/Main/Desktop/123/video1.avi", "C:/Users/Main/Desktop/123/video2.avi")); // Замените на действительные пути при необходимости
}

// Тест для обнаружения изменений между двумя кадрами
BOOST_AUTO_TEST_CASE(detect_changes_test) {

    Processing processing("C:/Users/Main/Desktop/123/video1.avi", "C:/Users/Main/Desktop/123/video2.avi"); // Псевдопути

    cv::Mat frame1 = createTestFrame(100); // Серый кадр со значением 100
    cv::Mat frame2 = createTestFrame(200); // Серый кадр со значением 200
    cv::Mat combinedFrame;
    cv::Mat thresh; // Создаем Mat для параметра thresh

    // Симулируем чтение кадров (можно манипулировать внутренним состоянием здесь)
    processing.detectedChanges(combinedFrame, thresh); // Передаем переменную thresh

    // Проверяем, имеет ли комбинированный кадр ожидаемые свойства
    BOOST_CHECK(!combinedFrame.empty()); // Убедимся, что комбинированный кадр не пуст
    BOOST_CHECK_EQUAL(combinedFrame.rows, frame1.rows); // Проверяем, совпадает ли высота
    BOOST_CHECK_EQUAL(combinedFrame.cols * 2, frame1.cols); // Проверяем, удвоена ли ширина
}

// Тест для инициализации фона
BOOST_AUTO_TEST_CASE(background_initialization_test) {
    Processing processing("C:/Users/Main/Desktop/123/video1.avi", "C:/Users/Main/Desktop/123/video2.avi");
    cv::Mat frame1 = createTestFrame(100); // Серый кадр
    cv::Mat combinedFrame, thresh;

    processing.detectedChanges(combinedFrame, thresh); // Обрабатываем первый кадр

    // Проверяем, был ли инициализирован фон
    BOOST_CHECK(!processing.background.empty());
    BOOST_CHECK_EQUAL(processing.background.type(), CV_32F); // Проверяем тип фона
}

BOOST_AUTO_TEST_CASE(background_processing_test) {
    Processing processing("C:/Users/Main/Desktop/123/video1.avi", "C:/Users/Main/Desktop/123/video2.avi");

    // Создаем тестовые кадры
    cv::Mat frame1 = createTestFrame(100); // Серый кадр
    cv::Mat combinedFrame, thresh;

    // Обрабатываем первый кадр
    processing.detectedChanges(combinedFrame, thresh);

    // Проверяем, что фон был инициализирован
    BOOST_CHECK(!processing.background.empty());
    BOOST_CHECK_EQUAL(processing.background.type(), CV_32F);
}
BOOST_AUTO_TEST_SUITE_END()
//=====================================================================================================

// Тесты для класса TrackingBox ===================================================================================
BOOST_AUTO_TEST_SUITE(TrackingBoxTests)

// Проверка, что конструктор корректно инициализирует координаты и размеры бокса.
BOOST_AUTO_TEST_CASE(ConstructorTest) {
    TrackingBox box(10, 20, 30, 40);

    BOOST_CHECK_EQUAL(std::get<0>(box.shape()), 10);
    BOOST_CHECK_EQUAL(std::get<1>(box.shape()), 20);
    BOOST_CHECK_EQUAL(std::get<2>(box.shape()), 30);
    BOOST_CHECK_EQUAL(std::get<3>(box.shape()), 40);
}

// Проверяет, что метод rectangle возвращает правильные значения для созданного бокса.
BOOST_AUTO_TEST_CASE(RectangleTest) {
    TrackingBox box(10, 20, 30, 40);
    cv::Rect rect = box.rectangle();

    BOOST_CHECK_EQUAL(rect.x, 10);
    BOOST_CHECK_EQUAL(rect.y, 20);
    BOOST_CHECK_EQUAL(rect.width, 30);
    BOOST_CHECK_EQUAL(rect.height, 40);
}

// Проверяет, что метод trackingCreation обновляет список трекеров, добавляя новые детекции.
BOOST_AUTO_TEST_CASE(TrackingCreationTest) {
    std::vector<TrackingBox> detections = { TrackingBox(10, 20, 30, 40), TrackingBox(50, 60, 30, 40) };
    std::vector<TrackingBox> trackers = { TrackingBox(10, 20, 30, 40) };

    auto updated_trackers = TrackingBox::trackingCreation(detections, trackers);

    BOOST_CHECK_EQUAL(updated_trackers.size(), 2); // 1 трекер + 1 новая детекция
}

// Тест на сравнение гистограмм
BOOST_AUTO_TEST_CASE(compare_histograms_test) {
    // Создаем некоторые фиктивные гистограммы для тестирования
    cv::Mat hist1 = cv::Mat::zeros(256, 1, CV_32F);
    cv::Mat hist2 = cv::Mat::zeros(256, 1, CV_32F);

    // Заполняем гистограммы некоторыми значениями
    hist1.at<float>(50) = 1.0f; 
    hist2.at<float>(50) = 0.8f;  

    // Создаем два экземпляра TrackingBox
    TrackingBox box1(10, 20, 30, 40);
    TrackingBox box2(50, 60, 30, 40);

    // Симулируем начальное состояние del_hists
    TrackingBox::del_hists[box1.getId()].push_back(hist1);
    TrackingBox::del_hists[box1.getId()].push_back(hist2);

    // Создаем новую гистограмму для сравнения
    cv::Mat new_hist = cv::Mat::zeros(256, 1, CV_32F);
    new_hist.at<float>(50) = 0.9f;

    // Выполняем сравнение гистограмм
    auto results = TrackingBox::compareHistograms(new_hist);

    // Проверяем, содержат ли результаты ожидаемый ID и значения корреляции
    BOOST_CHECK_EQUAL(results[box1.getId()].size(), 2); 
}

BOOST_AUTO_TEST_SUITE_END()
//====================================================================================================

// Тесты для Intersection ===================================================================================
BOOST_AUTO_TEST_SUITE(IOUTests)
// Тест на проверку вычисления IoU для полностью перекрывающихся прямоугольников
BOOST_AUTO_TEST_CASE(intersection_over_union_full_overlap_test) {
    IntersectOverUnion iouCalculator;

    cv::Rect rectA(0, 0, 2, 2); // Прямоугольник A
    cv::Rect rectB(0, 0, 2, 2); // Прямоугольник B (полное совпадение)

    double iou = iouCalculator.intersectionOverUnion(rectA, rectB);

    BOOST_CHECK_CLOSE(iou, 1.0, 0.0001); // Полное перекрытие должно давать IoU = 1
}

// Тест на проверку вычисления IoU для непересекающихся прямоугольников
BOOST_AUTO_TEST_CASE(intersection_over_union_no_overlap_test) {
    IntersectOverUnion iouCalculator;

    cv::Rect rectA(0, 0, 2, 2); // Прямоугольник A
    cv::Rect rectB(3, 3, 2, 2); // Прямоугольник B (не пересекается)

    double iou = iouCalculator.intersectionOverUnion(rectA, rectB);

    BOOST_CHECK_CLOSE(iou, 0.0, 0.0001); // Непересекающиеся прямоугольники должны давать IoU = 0
}

// Тест на сопоставление трекеров и детекций
BOOST_AUTO_TEST_CASE(match_test) {
    IntersectOverUnion iouCalculator(0.5); // Устанавливаем порог IoU 0.5

    // Создаем матрицу IoU для 2 трекеров и 2 детекций
    cv::Mat IoU = (cv::Mat_<double>(2, 2) << 0.6, 0.2, 0.3, 0.7);

    // Создаем трекеры и детекции
    std::vector<TrackingBox> trackers = { TrackingBox(0, 0, 2, 2), TrackingBox(3, 3, 2, 2) };
    std::vector<TrackingBox> detections = { TrackingBox(1, 1, 2, 2), TrackingBox(4, 4, 2, 2) };

    // Выполняем сопоставление
    auto result = iouCalculator.match(IoU, trackers, detections);

    // Проверяем результаты
    const auto& matches = std::get<0>(result);
    const auto& unmatched_detections = std::get<1>(result);
    const auto& unmatched_trackers = std::get<2>(result);

    BOOST_CHECK_EQUAL(matches.size(), 2); // Должно быть 2 совпадения
    BOOST_CHECK_EQUAL(unmatched_detections.size(), 0); // Должно быть 0 несопоставленных детекций
    BOOST_CHECK_EQUAL(unmatched_trackers.size(), 0); // Должно быть 0 несопоставленных трекеров
}

// Тест на сопоставление с несопоставленными трекерами
BOOST_AUTO_TEST_CASE(match_with_unmatched_test) {
    IntersectOverUnion iouCalculator(0.5); // Устанавливаем порог IoU 0.5

    // Создаем матрицу IoU для 3 трекеров и 2 детекций
    cv::Mat IoU = (cv::Mat_<double>(3, 2) << 0.6, 0.2, 0.3, 0.7, 0.1, 0.1);

    // Создаем трекеры и детекции
    std::vector<TrackingBox> trackers = { TrackingBox(0, 0, 2, 2), TrackingBox(3, 3, 2, 2), TrackingBox(6, 6, 2, 2) };
    std::vector<TrackingBox> detections = { TrackingBox(1, 1, 2, 2), TrackingBox(4, 4, 2, 2) };

    // Выполняем сопоставление
    auto result = iouCalculator.match(IoU, trackers, detections);

    // Проверяем результаты
    const auto& matches = std::get<0>(result);
    const auto& unmatched_detections = std::get<1>(result);
    const auto& unmatched_trackers = std::get<2>(result);

    BOOST_CHECK_EQUAL(matches.size(), 2); // Должно быть 2 совпадения
    BOOST_CHECK_EQUAL(unmatched_detections.size(), 0); // Должно быть 0 несопоставленных детекций
    BOOST_CHECK_EQUAL(unmatched_trackers.size(), 1); // Должно быть 1 несопоставленный трекер
}

BOOST_AUTO_TEST_SUITE_END()
//=========================================================================================================

// Тесты для алгоритма Венгерского (Hungarian Algorithm) ===================================================================================
BOOST_AUTO_TEST_SUITE(HungarianTest)

// Тест для матрицы 2x2
BOOST_AUTO_TEST_CASE(hungarian_algorithm_2x2_test) {
    HungarianAlgorithm hungarian;
    std::vector<std::vector<double>> distMatrix = {
        {4, 2},
        {2, 3}
    };
    std::vector<int> assignment;

    double cost = hungarian.Solve(distMatrix, assignment);

    BOOST_CHECK_CLOSE(cost, 4.0, 0.0001); // Минимальная стоимость должна быть 4
    BOOST_CHECK_EQUAL(assignment[0], 1); // Первая задача назначена второму работнику
    BOOST_CHECK_EQUAL(assignment[1], 0); // Вторая задача назначена первому работнику
}

// Тест для пустой матрицы расстояний
BOOST_AUTO_TEST_CASE(hungarian_algorithm_empty_matrix_test) {
    HungarianAlgorithm hungarian;
    std::vector<std::vector<double>> distMatrix = {};
    std::vector<int> assignment;

    double cost = hungarian.Solve(distMatrix, assignment);

    BOOST_CHECK_CLOSE(cost, 0.0, 0.0001); // Стоимость должна быть 0
    BOOST_CHECK(assignment.empty()); // Назначение должно быть пустым
}

// Тест для обработки матрицы расстояний с одинаковыми значениями
BOOST_AUTO_TEST_CASE(hungarian_algorithm_identical_values_test) {
    HungarianAlgorithm hungarian;
    std::vector<std::vector<double>> distMatrix = {
        {1, 1},
        {1, 1}
    };
    std::vector<int> assignment;

    double cost = hungarian.Solve(distMatrix, assignment);

    BOOST_CHECK_CLOSE(cost, 2.0, 0.0001); // Минимальная стоимость должна быть 2
    BOOST_CHECK_EQUAL(assignment[0], 0); // Первая задача назначена первому работнику
    BOOST_CHECK_EQUAL(assignment[1], 1); // Вторая задача назначена второму работнику
}

BOOST_AUTO_TEST_SUITE_END()
