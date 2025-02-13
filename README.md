# ProjTranfCpp

В файле описаны классы - методы, назначение функций, входные и выходные данные и принцип их работы.

## Processing.cpp
 Предназначен для обработки видео с целью обнаружения изменений между кадрами. Метод позволяет обнаруживать изменения между кадрами двух видео, устанавливая фон и выделяя контуры движущихся объектов.  
  1. Метод Processing - Инициализация объекта класса Processing, открытие потоков видео для дальнейшей обработки.
     - 1.1 Входные данные:
        - const std::string& fileName1: Путь к первому видеофайлу.
        -	const std::string& fileName2: Путь ко второму видеофайлу.
     - 1.2 Алгоритм:
        -	Происходит инициализация переменных isFrame и background.
        -	Открываются два потока видео с помощью cv::VideoCapture.
        -	Если хотя бы один поток не удалось открыть, выбрасываем исключение с описанием ошибки.
  2. Метод detectedChanges - Обнаружение изменений между текущими кадрами двух видео.
      - 2.1 Входные данные:
           - cv::Mat& frame: Ссылка на изображение, в которое будет записан объединенный кадр.
           -	cv::Mat& thresh: Ссылка на изображение, в котором будет храниться результат пороговой обработки.

       - 2.2 Алгоритм:
           - 	Инициализируем фоновое изображение, если оно пустое, копируем текущий серый кадр в background.
           - 	Если фоновое изображение уже инициализировано, обновляем его с помощью cv::accumulateWeighted.
           - 	Преобразуем фон в 8-битный формат.
           - 	Вычисляем абсолютную разность между фоном и текущим кадром.
           - 	Применяем пороговую обработку для выделения изменений.
           - 	Выполняем дилатацию для улучшения контуров.
           - 	Применяем гауссово размытие для снижения шумов.
           - 	Находим контуры в обработанном изображении.
           - 	Создаем боксы на основе обнаруженных контуров с помощью TrackingBox::createBoxes.
           - 	Копируем результат пороговой обработки в grayFrame.
           - Возвращаем вектор боксов (TrackingBox).

## HungarianAlgorithm.cpp 
 Код реализует Венгерский алгоритма, предназначенный для решения задачи назначения. Задача назначения —задача, где необходимо сопоставить элементы из одного множества с элементами из другого множества, минимизируя общую стоимость.
 1. Метод Solve - Основной метод, который решает задачу назначения.
    - 1.1 Входные данные:
         - DistMatrix: Двумерный вектор (матрица) расстояний (стоимостей) между элементами двух множеств.
         - Assignment: Вектор, в который будет записан результат назначения.
    - 1.2 Выходные данные:
         - Возвращает общую стоимость оптимального назначения.
    - 1.3 Алгоритм:
        - Проверяет, пуста ли матрица расстояний.
        -	Преобразует двумерную матрицу в одномерный вектор для удобства обработки.
        - Вызывает метод assignmentoptimal для нахождения оптимального назначения.
        - Возвращает стоимость назначения.
  2. Метод assignmentoptimal - Находит оптимальное назначение с минимальной стоимостью.
    - 2.1 Входные данные:
       - assignment: Вектор для хранения результата назначения.
       -	cost: Указатель на переменную, в которую будет записана общая стоимость.
       -	distMatrixIn: Одномерный вектор, представляющий матрицу расстояний.
       -	nOfRows, nOfColumns: Количество строк и столбцов в матрице.
    - 2.2 Выходные данные:
        	-Вектор, в который записывается результат назначения. Каждый элемент assignment[i] содержит индекс задачи, назначенной i-му работнику. Если задача не назначена, значение будет -1.

         - Указатель на переменную, в которую записывается общая стоимость назначения
     - 2.3 Алгоритм:
         - Вычитает минимальные значения из каждой строки матрицы, чтобы упростить задачу.
         -	Отмечает "звезды" в матрице (нулевые элементы, которые могут быть частью оптимального назначения).
         -	Если количество "звезд" равно минимальному измерению (числу строк или столбцов), строит вектор назначения.
         -	В противном случае вызывает метод starMethod для дальнейшей обработки.
         -	Вычисляет итоговую стоимость назначения.




     
