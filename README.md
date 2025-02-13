# ProjTranfCpp

В файле описаны классы - методы, назначение функций, входные и выходные данные и принцип их работы.

## Processing.cpp
 Предназначен для обработки видео с целью обнаружения изменений между кадрами. Метод позволяет обнаруживать изменения между кадрами двух видео, устанавливая фон и выделяя контуры движущихся объектов.  
 ### 1. Метод Processing - Инициализация объекта класса Processing, открытие потоков видео для дальнейшей обработки.
     - 1.1 Входные данные:
        - const std::string& fileName1: Путь к первому видеофайлу.
        -	const std::string& fileName2: Путь ко второму видеофайлу.
     - 1.2 Алгоритм:
        -	Происходит инициализация переменных isFrame и background.
        -	Открываются два потока видео с помощью cv::VideoCapture.
        -	Если хотя бы один поток не удалось открыть, выбрасываем исключение с описанием ошибки.
 ### 2. Метод detectedChanges - Обнаружение изменений между текущими кадрами двух видео.
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
           -  Возвращаем вектор боксов (TrackingBox).

## HungarianAlgorithm.cpp 
 Код реализует Венгерский алгоритма, предназначенный для решения задачи назначения. Задача назначения —задача, где необходимо сопоставить элементы из одного множества с элементами из другого множества, минимизируя общую стоимость.
### 1. Метод Solve - Основной метод, который решает задачу назначения.
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

### 2. Метод assignmentoptimal - Находит оптимальное назначение с минимальной стоимостью.
    - 2.1 Входные данные:
       - assignment: Вектор для хранения результата назначения.
       -	cost: Указатель на переменную, в которую будет записана общая стоимость.
       -	distMatrixIn: Одномерный вектор, представляющий матрицу расстояний.
       -	nOfRows, nOfColumns: Количество строк и столбцов в матрице.

    - 2.2 Выходные данные:
       
       - Вектор, в который записывается результат назначения. Каждый элемент assignment[i] содержит индекс задачи, назначенной i-му работнику. Если задача не назначена, значение будет -1.
       - Указатель на переменную, в которую записывается общая стоимость назначения

    - 2.3 Алгоритм:
        - Вычитает минимальные значения из каждой строки матрицы, чтобы упростить задачу.
        -	Отмечает "звезды" в матрице (нулевые элементы, которые могут быть частью оптимального назначения).
        -	Если количество "звезд" равно минимальному измерению (числу строк или столбцов), строит вектор назначения.
        -	В противном случае вызывает метод starMethod для дальнейшей обработки.
        -	Вычисляет итоговую стоимость назначения.
 
 ### 3. Метод buildassignmentvector - Строит вектор назначения на основе матрицы "звезд".
     - 3.1 Входные данные:
          -	assignment: Вектор для хранения результата назначения.
          -	starMatrix: Матрица "звезд", где true указывает на возможное назначение.
          -	nOfRows, nOfColumns: Количество строк и столбцов в матрице.
     - 3.2 Выходные данные:
          - Вектор, в который записывается результат назначения. Каждый элемент assignment[i] содержит индекс задачи, назначенной i-му работнику. Если задача не назначена,  значение будет -1.
     - 3.3 Алгоритм:
          - Проходит по матрице "звезд" и заполняет вектор назначения
### 4. Метод computeassignmentcost - Вычисляет общую стоимость назначения.
      - 4.1 Входные данные:
          -	assignment: Вектор назначения.
          - cost: Указатель на переменную, в которую будет записана стоимость.
          -	distMatrix: Матрица расстояний.
          - nOfRows: Количество строк в матрице.

     - 4.2 Выходные данные:
          - Указатель на переменную, в которую записывается общая стоимость назначения.
     - 4.3 Алгоритм:
          - Суммирует стоимости назначений, используя вектор назначения и матрицу расстояний.
### 5. Метод starMethod - Реализует "звездный метод" для нахождения оптимального назначения.
     
     - 5.1 Входные данные:
          -	assignment: Вектор для хранения результата назначения.
          -	distMatrix: Матрица расстояний.
          -	starMatrix: Матрица "звезд".
          -	primeMatrix: Матрица для пометок.
          -	coveredColumns, coveredRows: Векторы, указывающие, какие столбцы и строки закрыты.
          -	nOfRows, nOfColumns: Количество строк и столбцов в матрице.
          -	minDim: Минимальное измерение (число строк или столбцов).
   
     - 5.2 Алгоритм:
          -	Ищет нулевые элементы в не закрытых строках и столбцах.
          -	Если нулевой элемент найден, помечает его и проверяет, есть ли в этой строке "звезда".
          -	Если "звезда" найдена, открывает соответствующий столбец.
          -	Если нулевые элементы не найдены, находит минимальное значение в не закрытых строках и столбцах.
          -	Вычитает минимальное значение из не закрытых строк и добавляет его к не закрытым столбцам.
          -	Рекурсивно вызывает себя для продолжения поиска оптимального назначения.

## TrackingBox.cpp - Класс TrackingBox предназначен для реализации системы отслеживания объектов на видео. Он создает и обновляет трекеры для обнаруженных объектов, используя методы сопоставления, такие как IoU (Intersection over Union) и венгерский алгоритм для оптимизации соответствий между трекерами и обнаруженными объектами. Класс также управляет гистограммами, которые используются для определения схожести объектов.

### 1. Метод TrackingBox - Инициализирует объект трекера с заданными координатами и размерами, а также уникальным идентификатором. 
    - 1.1 Входные данные:

       - int x, int y: координаты верхнего левого угла прямоугольника.

       - int w, int h: ширина и высота прямоугольника.
       
    - 1.2 Выходные данные:

       - Объект класса TrackingBox, инициализированный с указанными координатами и размерами.

    - 1.3 Алгоритм: 
       - Инициализирует координаты и размеры трекера.

       - Присваивает уникальный ID трекеру, увеличивая счетчик id_counter.

### 2. Метод trackingCreation - Метод создает или обновляет трекеры на основе обнаруженных объектов. Если трекеры отсутствуют, возвращаются обнаруженные объекты. Если обнаруженные объекты отсутствуют, трекеры удаляются. Для сопоставления объектов используется матрица IoU (Intersection over Union) и алгоритм Венгера. 
    
    - 1.1 Входные данные:
       - std::vector<TrackingBox>& detections: список обнаруженных объектов.

       - std::vector<TrackingBox>& trackers: список текущих трекеров.
       
    - 1.2 Выходные данные:

       - std::vector<TrackingBox>: обновленный список трекеров..

    - 1.3 Алгоритм: 
       
       - Если трекеры отсутствуют, возвращает список обнаруженных объектов.
        
       - Если обнаруженные объекты отсутствуют, удаляет все трекеры, сохраняя их гистограммы в del_hists.
        
       - Создает матрицу IoU (Intersection over Union) для трекеров и обнаруженных объектов.
        
       - Сопоставляет трекеры и обнаруженные объекты с помощью Венгерского алгоритма.
        
       - Обновляет ID для сопоставленных объектов.
        
       - Добавляет несопоставленные обнаруженные объекты в список трекеров.
        
       - Удаляет несопоставленные трекеры, сохраняя их гистограммы в del_hists.
        
       - Возвращает обновленный список трекеров.



     
