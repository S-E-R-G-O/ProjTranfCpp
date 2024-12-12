#include "pch.h" 
#include "C:/Users/Main/source/repos/transfCpp/transfCpp/Processing.cpp" // Подключение файла Processing.cpp
#include "C:/Users/Main/source/repos/transfCpp/transfCpp/TrackingBox.cpp" // Подключение файла TrackingBox.cpp
#include "C:/Users/Main/source/repos/transfCpp/transfCpp/IntersertoinOverUnion.cpp" // Подключение файла IntersectionOverUnion.cpp
#include "C:/Users/Main/source/repos/transfCpp/transfCpp/HungarianAlgorithm.cpp" // Подключение файла HungarianAlgorithm.cpp

// Тестовый класс, который наследует от Processing, чтобы проверить его методы
class ProcessingTest : public Processing {
public:
    ProcessingTest(const std::string& fileName1, const std::string& fileName2)
        : Processing(fileName1, fileName2) {} // Конструктор, который инициализирует базовый класс

    using Processing::processingFrame; // Делаем метод processingFrame доступным
};

// Тесты для проверки открытия видеопотока класса Processing
TEST(Processing, ValidVideoFiles) {
    // Предполагаем, что указанные пути действительны и видеофайлы существуют
    const std::string validFile1 = "C:/Users/Main/Desktop/123/video1.avi"; // Путь к первому видео
    const std::string validFile2 = "C:/Users/Main/Desktop/123/video2.avi"; // Путь ко второму видео

    // Создаем объект Processing и ожидаем, что исключений не будет
    EXPECT_NO_THROW(Processing processing(validFile1, validFile2));
}

TEST(Processing, InvalidVideoFiles) {
    // Указываем недействительные пути к видеофайлам
    const std::string invalidFile1 = "C:/InvalidPath/video1.avi"; // Недействительный путь к первому видео
    const std::string invalidFile2 = "C:/InvalidPath/video2.avi"; // Недействительный путь ко второму видео

    // Ожидаем, что при попытке создать объект Processing будет выброшено исключение
    EXPECT_THROW(Processing processing(invalidFile1, invalidFile2), std::runtime_error);
}

// Тесты для метода processingFrame класса Processing
TEST(Processing, ProcessingFrame) {
    // Создаем временный объект Processing с двумя видеофайлами
    const std::string validFile1 = "C:/Users/Main/Desktop/123/video1.avi"; // Путь к первому видео
    const std::string validFile2 = "C:/Users/Main/Desktop/123/video2.avi"; // Путь ко второму видео

    Processing processing(validFile1, validFile2);

    // Создаем пустые кадры для тестирования
    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3); // Пустое цветное изображение
    cv::Mat grayFrame = cv::Mat::zeros(480, 640, CV_8UC1); // Пустое серое изображение

    // Вызываем метод processingFrame и получаем обнаруженные боксы
    std::vector<TrackingBox> boxes = processing.processingFrame(frame, grayFrame);

    // Ожидаем, что метод вернет пустой вектор, так как входные изображения пустые
    EXPECT_TRUE(boxes.empty());
}

// Тест для функции detectedChanges
TEST(Processing, DetectedChanges) {
    const std::string validFile1 = "C:/Users/Main/Desktop/123/video1.avi"; // Путь к первому видео
    const std::string validFile2 = "C:/Users/Main/Desktop/123/video2.avi"; // Путь ко второму видео

    ProcessingTest processing(validFile1, validFile2); // Создание тестового объекта Processing

    cv::Mat frame, thresh; // Создание матриц для кадра и порога
    // Предполагаем, что в видеопотоках есть кадры
    EXPECT_NO_THROW({
        std::vector<TrackingBox> detected = processing.detectedChanges(frame, thresh);
        EXPECT_TRUE(detected.empty()); // Ожидаем, что обнаруженные изменения пусты
        });
}

//==========================TRACKINGBOX.CPP==============================================
// Проверка инициализации параметров бокса
TEST(TrackingBox, Constructor) {
    TrackingBox box(10, 20, 30, 40); // Создание объекта TrackingBox
    EXPECT_EQ(box.shape(), std::make_tuple(10, 20, 30, 40)); // Проверка корректности инициализации
}

// Проверка, что метод drawBoxes корректно отрисовывает боксы на изображении и не вызывает исключений
TEST(TrackingBox, DrawBoxes) {
    cv::Mat frame = cv::Mat::zeros(100, 100, CV_8UC3); // Создание пустого изображения
    TrackingBox box1(10, 10, 30, 30); // Первый бокс
    TrackingBox box2(50, 50, 20, 20); // Второй бокс

    std::vector<TrackingBox> boxes = { box1, box2 }; // Вектор боксов

    // Ожидаем, что метод drawBoxes не вызовет исключений
    EXPECT_NO_THROW(TrackingBox::drawBoxes(frame, boxes));
}

// Проверка, что метод processHistogram не вызывает исключений при обработке пустого изображения
TEST(TrackingBox, ProcessHistogram) {
    cv::Mat frame = cv::Mat::zeros(100, 100, CV_8UC3); // Создание пустого изображения
    TrackingBox box(10, 10, 30, 30); // Создание объекта TrackingBox

    // Процесс обработки гистограммы не должен вызывать исключений
    EXPECT_NO_THROW(box.processHistogram(frame));
}

// Проверка, что метод rectangle возвращает правильный прямоугольник OpenCV
TEST(TrackingBox, Rectangle) {
    TrackingBox box(10, 20, 30, 40); // Создание объекта TrackingBox
    cv::Rect rect = box.rectangle(); // Получение прямоугольника

    EXPECT_EQ(rect.x, 10); // Проверка координаты x
    EXPECT_EQ(rect.y, 20); // Проверка координаты y
    EXPECT_EQ(rect.width, 30); // Проверка ширины
    EXPECT_EQ(rect.height, 40); // Проверка высоты
}

// Тест для проверки корректности функции meanVal
TEST(TrackingBox, PrintMeanval) {
    // Создание объекта TrackingBox
    TrackingBox box(10, 20, 30, 40);

    // Создание тестовых гистограмм
    cv::Mat b_hist = cv::Mat::zeros(256, 1, CV_64F); // Гистограмма для синего канала
    cv::Mat g_hist = cv::Mat::zeros(256, 1, CV_64F); // Гистограмма для зеленого канала
    cv::Mat r_hist = cv::Mat::zeros(256, 1, CV_64F); // Гистограмма для красного канала

    // Установка значений для гистограмм
    for (int i = 0; i < 256; i++) {
        b_hist.at<double>(i) = i / 255.0; // Значения для синего канала
        g_hist.at<double>(i) = (255 - i) / 255.0; // Значения  зеленого канала
        r_hist.at<double>(i) = 0.5; // Значения  красного канала
    }

    
    std::ostringstream buffer; // Создание буфера для вывода
    std::streambuf* oldCoutBuf = std::cout.rdbuf(buffer.rdbuf()); // Сохраняем старый буфер

   
    box.printMeanval(b_hist, g_hist, r_hist, "top");

   
    std::cout.rdbuf(oldCoutBuf);

    // Проверка, что вывод содержит ожидаемое значение
    std::string output = buffer.str();
    EXPECT_NE(output.find("Mean color values in the top half of the tracking box:"), std::string::npos);
    EXPECT_NE(output.find("Average: "), std::string::npos);
}

//==========================IntersectioOverUnion================

// Тест для конструктора IntersectOverUnion
TEST(IntersectOverUnion, Constructor) {
   
    IntersectOverUnion iou;
    EXPECT_NO_THROW(iou = IntersectOverUnion()); 

    // Тест с пользовательским порогом IoU
    double custom_threshold = 0.5;
    EXPECT_NO_THROW(IntersectOverUnion iou_custom(custom_threshold));
}

// Тест для метода intersectionOverUnion
TEST(IntersectOverUnion, IntersectionOverUnion) {
    cv::Rect rectA(0, 0, 10, 10); // Прямоугольник A
    cv::Rect rectB(5, 5, 10, 10); // Прямоугольник B
    double iou_value = IntersectOverUnion::intersectionOverUnion(rectA, rectB);

    // Ожидаемое значение IoU
    double expected_iou = 0.142857; // Площадь пересечения 25, площадь объединения 175
    EXPECT_NEAR(iou_value, expected_iou, 1e-5); // Допускаем небольшую погрешность

    // Тест без пересечения
    cv::Rect rectC(20, 20, 10, 10); // Прямоугольник C
    double iou_value_no_intersection = IntersectOverUnion::intersectionOverUnion(rectA, rectC);
    EXPECT_EQ(iou_value_no_intersection, 0.0); // IoU должен быть 0
}

// Тест для метода match
TEST(IntersectOverUnion, Match) {
    // Создание матрицы IoU
    cv::Mat IoU = (cv::Mat_<double>(2, 2) << 0.5, 0.1, 0.4, 0.7);

    // Создание трекеров и детекций
    TrackingBox tracker1(0, 0, 10, 10); // Первый трекер
    TrackingBox tracker2(10, 10, 10, 10); // Второй трекер
    TrackingBox detection1(5, 5, 10, 10); // Первая детекция
    TrackingBox detection2(15, 15, 10, 10); // Вторая детекция

    std::vector<TrackingBox> trackers = { tracker1, tracker2 }; // Вектор трекеров
    std::vector<TrackingBox> detections = { detection1, detection2 }; // Вектор детекций

    IntersectOverUnion iouMatcher(0.3); // Установка порога IoU

    // Выполнение сопоставления
    auto [matches, unmatched_detections, unmatched_trackers] = iouMatcher.match(IoU, trackers, detections);

    // Проверка совпадений
    EXPECT_EQ(matches.size(), 2); // Должно быть 2 совпадения
    EXPECT_EQ(matches[0][0], 0); // Трекер 0 совпадает с детекцией 0
    EXPECT_EQ(matches[0][1], 0); // Детекция 0 совпадает с трекером 0
    EXPECT_EQ(matches[1][0], 1); // Трекер 1 совпадает с детекцией 1
    EXPECT_EQ(matches[1][1], 1); // Детекция 1 совпадает с трекером 1

    // Проверка неподходящих трекеров и детекций
    EXPECT_TRUE(unmatched_trackers.empty()); // Нет неподходящих трекеров
    EXPECT_TRUE(unmatched_detections.empty()); // Нет неподходящих детекций
}

// Тест для совпадений с неподходящими случаями
TEST(IntersectOverUnion, MatchWithUnmatched) {
    
    cv::Mat IoU = (cv::Mat_<double>(2, 2) << 0.1, 0.1, 0.1, 0.1);

    // Создание трекеров и детекций
    TrackingBox tracker1(0, 0, 10, 10);
    TrackingBox tracker2(10, 10, 10, 10);
    TrackingBox detection1(15, 15, 10, 10);
    TrackingBox detection2(20, 20, 10, 10);

    std::vector<TrackingBox> trackers = { tracker1, tracker2 };
    std::vector<TrackingBox> detections = { detection1, detection2 };

    IntersectOverUnion iouMatcher(0.3); // Установка порога IoU

    // Выполнение сопоставления
    auto [matches, unmatched_detections, unmatched_trackers] = iouMatcher.match(IoU, trackers, detections);

    // Проверка совпадений
    EXPECT_TRUE(matches.empty()); 

    // Проверка неподходящих трекеров и детекций
    EXPECT_EQ(unmatched_trackers.size(), 2); 
    EXPECT_EQ(unmatched_detections.size(), 2); 
}

//=========================HungarianAlgoritm=========================
// Тест для конструктора и деструктора HungarianAlgorithm
TEST(HungarianAlgorithm, ConstructorDestructor) {
    EXPECT_NO_THROW(HungarianAlgorithm algorithm); 
}

// Тест случая когда строк больше столбцов
TEST(HungarianAlgorithm, SolveUnbalancedMatrix) {
    HungarianAlgorithm algorithm;

    
    std::vector<std::vector<double>> DistMatrix = {
        {1.0, 2.0},
        {4.0, 6.0},
        {7.0, 8.0}
    };

    std::vector<int> Assignment; 
    double cost; 

  
    EXPECT_NO_THROW(cost = algorithm.Solve(DistMatrix, Assignment));

    
    EXPECT_EQ(Assignment.size(), 3);
    EXPECT_GE(cost, 0.0); 
}

// Тест для квадратичной нулевой матрицы 
TEST(HungarianAlgorithm, SolveZeroMatrix) {
    HungarianAlgorithm algorithm;

   
    std::vector<std::vector<double>> DistMatrix = {
        {0.0, 0.0},
        {0.0, 0.0}
    };

    std::vector<int> Assignment; 
    double cost; 

    //
    EXPECT_NO_THROW(cost = algorithm.Solve(DistMatrix, Assignment));

   
    EXPECT_EQ(Assignment.size(), 2); 
    EXPECT_EQ(cost, 0.0); 
}

// Тест для решения с квадратной матрицей
TEST(HungarianAlgorithm, SolveSquareMatrix) {
    HungarianAlgorithm algorithm;

    // Определяем квадратную матрицу расстояний
    std::vector<std::vector<double>> DistMatrix = {
        {4.0, 2.0, 8.0},
        {2.0, 4.0, 6.0},
        {6.0, 1.0, 3.0}
    };

    std::vector<int> Assignment; 
    double cost; 

    EXPECT_NO_THROW(cost = algorithm.Solve(DistMatrix, Assignment));

    
    EXPECT_EQ(Assignment.size(), 3);
    EXPECT_GE(cost, 0.0); 
}

// Тест для случая при котором столбцов больше строк
TEST(HungarianAlgorithm, SolveMoreColumnsThanRows) {
    HungarianAlgorithm algorithm;

    std::vector<std::vector<double>> DistMatrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    std::vector<int> Assignment; 
    double cost; 

    
    EXPECT_NO_THROW(cost = algorithm.Solve(DistMatrix, Assignment));

   
    EXPECT_EQ(Assignment.size(), 2); 
    EXPECT_GE(cost, 0.0); 
}

// Тест для более сложной матрицы
TEST(HungarianAlgorithm, SolveComplexMatrix) {
    HungarianAlgorithm algorithm;

    // Определяем более сложную матрицу расстояний
    std::vector<std::vector<double>> DistMatrix = {
        {10.0, 19.0, 8.0, 15.0},
        {10.0, 18.0, 7.0, 17.0},
        {13.0, 16.0, 9.0, 14.0},
        {12.0, 19.0, 20.0, 8.0}
    };

    std::vector<int> Assignment; 
    double cost; 

   
    EXPECT_NO_THROW(cost = algorithm.Solve(DistMatrix, Assignment));

    EXPECT_EQ(Assignment.size(), 4); 
    EXPECT_GE(cost, 0.0); 
}

// Функция для запуска всех тестов
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv); // Инициализация Google Test
    return RUN_ALL_TESTS(); // Запуск всех тестов
}
