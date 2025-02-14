#include "HungarianAlgorithm.h"
#include <numeric>
#include <algorithm>
#include <limits>
#include <cmath>

// Метод Solve решает задачу назначения, используя матрицу расстояний DistMatrix и возвращая оптимальное назначение.
double HungarianAlgorithm::Solve(std::vector<std::vector<double>>& DistMatrix, std::vector<int>& Assignment) {
    // Проверяем, пуста ли матрица расстояний.
    if (DistMatrix.empty() || DistMatrix[0].empty()) return 0.0;

    int nRows = DistMatrix.size(); // Количество строк
    int nCols = DistMatrix[0].size(); // Количество столбцов
    std::vector<double> distMatrixIn(nRows * nCols); // Одномерный вектор для хранения расстояний
    std::vector<int> assignment(nRows, -1); // Вектор для хранения назначения, инициализированный значением -1
    double cost = 0.0; // Переменная для хранения общей стоимости назначения

    // Копирование значений из двумерного вектора в одномерный
    for (int i = 0; i < nRows; ++i) {
        std::copy(DistMatrix[i].begin(), DistMatrix[i].end(), distMatrixIn.begin() + i * nCols);
    }

    // Вызов метода для нахождения оптимального назначения
    assignmentoptimal(assignment, &cost, distMatrixIn, nRows, nCols);
    Assignment = std::move(assignment); // Перемещаем назначение в выходной аргумент
    return cost; // Возвращаем стоимость назначения
}

// Метод assignmentoptimal находит оптимальное назначение с минимальной стоимостью
void HungarianAlgorithm::assignmentoptimal(std::vector<int>& assignment, double* cost, const std::vector<double>& distMatrixIn, int nOfRows, int nOfColumns) {
    std::vector<double> distMatrix = distMatrixIn; // Копируем входную матрицу расстояний
    std::vector<bool> coveredColumns(nOfColumns, false), coveredRows(nOfRows, false); // Векторы для отслеживания закрытых строк и столбцов
    std::vector<bool> starMatrix(nOfRows * nOfColumns, false); // Матрица звезд
    int minDim = std::min(nOfRows, nOfColumns); // Минимальное измерение (число строк или столбцов)
    *cost = 0.0; // Инициализация стоимости

    // Вычитаем минимальные значения из строк
    std::vector<double> rowMin(nOfRows, std::numeric_limits<double>::max());
    for (int i = 0; i < nOfRows; ++i) {
        for (int j = 0; j < nOfColumns; ++j) {
            rowMin[i] = std::min(rowMin[i], distMatrix[i * nOfColumns + j]);
        }
    }

    // Вычитание минимальных значений из строк
    for (int i = 0; i < nOfRows; ++i) {
        for (int j = 0; j < nOfColumns; ++j) {
            distMatrix[i * nOfColumns + j] -= rowMin[i];
        }
    }

    // Отмечаем звезды в матрице
    for (int row = 0; row < nOfRows; ++row) {
        for (int col = 0; col < nOfColumns; ++col) {
            // Если элемент равен нулю и столбец еще не закрыт
            if (fabs(distMatrix[row * nOfColumns + col]) < std::numeric_limits<double>::epsilon() && !coveredColumns[col]) {
                starMatrix[row * nOfColumns + col] = true; // Отмечаем звезду
                coveredColumns[col] = true;
                break;
            }
        }
    }

    // Если количество закрытых столбцов равно минимальному измерению, строим вектор назначения
    if (std::count(coveredColumns.begin(), coveredColumns.end(), true) == minDim) {
        buildassignmentvector(assignment, starMatrix, nOfRows, nOfColumns);
    }
    else {
        std::vector<bool> primeMatrix(nOfRows * nOfColumns, false); // Матрица для пометок
        starMethod(assignment, distMatrix, starMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim); // Запускаем звездный метод
    }

    // Вычисляем итоговую стоимость назначения
    computeassignmentcost(assignment, cost, distMatrixIn, nOfRows);
}

// Метод для построения вектора назначений на основе матрицы звезд
void HungarianAlgorithm::buildassignmentvector(std::vector<int>& assignment, const std::vector<bool>& starMatrix, int nOfRows, int nOfColumns) {
    for (int row = 0; row < nOfRows; ++row) {
        for (int col = 0; col < nOfColumns; ++col) {
            if (starMatrix[row * nOfColumns + col]) {
                assignment[row] = col; // Устанавливаем назначение для строки
                break;
            }
        }
    }
}

// Метод для вычисления стоимости назначения
void HungarianAlgorithm::computeassignmentcost(const std::vector<int>& assignment, double* cost, const std::vector<double>& distMatrix, int nOfRows) {
    *cost = std::accumulate(assignment.begin(), assignment.end(), 0.0, [&](double sum, int col) {
        return col >= 0 ? sum + distMatrix[col] : sum; // Суммируем стоимость назначений
        });
}

// Метод звездного метода для нахождения оптимального назначения
void HungarianAlgorithm::starMethod(std::vector<int>& assignment, std::vector<double>& distMatrix, std::vector<bool>& starMatrix, std::vector<bool>& primeMatrix, std::vector<bool>& coveredColumns, std::vector<bool>& coveredRows, int nOfRows, int nOfColumns, int minDim) {
    bool zerosFound; // Флаг, показывающий, найдены ли нули
    do {
        zerosFound = false; // Сброс флага перед каждой итерацией
        for (int col = 0; col < nOfColumns; ++col) {
            if (coveredColumns[col]) continue; // Пропускаем закрытые столбцы

            for (int row = 0; row < nOfRows; ++row) {
                if (coveredRows[row] || fabs(distMatrix[row * nOfColumns + col]) >= std::numeric_limits<double>::epsilon()) continue; // Пропускаем закрытые строки и ненулевые элементы

                primeMatrix[row * nOfColumns + col] = true; // Помечаем элемент
                coveredRows[row] = true; // Закрываем строку

                // Проверяем, есть ли звезда в этой строке
                for (int starCol = 0; starCol < nOfColumns; ++starCol) {
                    if (starMatrix[row * nOfColumns + starCol]) {
                        coveredColumns[starCol] = false; // Если нашли звезду, открываем столбец
                        zerosFound = true; // Устанавливаем флаг
                        break;
                    }
                }

                // Если нули не найдены, то ставим звезду
                if (!zerosFound) {
                    starMatrix[row * nOfColumns + col] = true;

                    // Перемещаем звезды и помечаем строки
                    for (int starCol = col; starCol < nOfColumns; ++starCol) {
                        for (int r = 0; r < nOfRows; ++r) {
                            if (starMatrix[r * nOfColumns + starCol]) {
                                starMatrix[r * nOfColumns + starCol] = false; // Убираем звезду
                                for (int primeCol = 0; primeCol < nOfColumns; ++primeCol) {
                                    if (primeMatrix[r * nOfColumns + primeCol]) {
                                        starMatrix[r * nOfColumns + primeCol] = true; // Ставим звезду
                                        break;
                                    }
                                }
                                break;
                            }
                        }
                    }

                    std::fill(coveredRows.begin(), coveredRows.end(), false); // Сбрасываем закрытые строки
                    starMethod(assignment, distMatrix, starMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, nOfRows); // Рекурсивный вызов
                    return; // Выходим из функции
                }
            }
        }
    } while (zerosFound); // Если нашли нули, продолжаем

    // Если нули не найдены, находим минимальное значение в не закрытых строках и столбцах
    double h = std::numeric_limits<double>::max();
    for (int row = 0; row < nOfRows; ++row) {
        if (coveredRows[row]) continue; // Пропускаем закрытые строки
        for (int col = 0; col < nOfColumns; ++col) {
            if (!coveredColumns[col]) {
                h = std::min(h, distMatrix[row * nOfColumns + col]); // Находим минимальное значение
            }
        }
    }

    // Вычитаем минимальное значение из не закрытых строк
    for (int row = 0; row < nOfRows; ++row) {
        if (!coveredRows[row]) {
            for (int col = 0; col < nOfColumns; ++col) {
                distMatrix[row * nOfColumns + col] += h; // Вычитаем h из строки
            }
        }
    }

    // Добавляем h к не закрытым столбцам
    for (int col = 0; col < nOfColumns; ++col) {
        if (!coveredColumns[col]) {
            for (int row = 0; row < nOfRows; ++row) {
                distMatrix[row * nOfColumns + col] -= h; // Добавляем h к столбцу
            }
        }
    }

    starMethod(assignment, distMatrix, starMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim); // Повторный вызов метода
}
