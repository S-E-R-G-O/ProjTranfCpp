#include "HungarianAlgorithm.h"
#include <numeric>
#include <algorithm>
#include <limits>
#include <cmath>

// Метод Solve решает задачу назначения, используя матрицу расстояний DistMatrix и возвращая оптимальное назначение.
double HungarianAlgorithm::Solve(std::vector<std::vector<double>>& DistMatrix, std::vector<int>& Assignment) {
    if (DistMatrix.empty() || DistMatrix[0].empty()) return 0.0;

    int nRows = DistMatrix.size();
    int nCols = DistMatrix[0].size();
    std::vector<double> distMatrixIn(nRows * nCols);
    std::vector<int> assignment(nRows, -1);
    double cost = 0.0;

    // Копирование значений из двумерного вектора в одномерный
    for (int i = 0; i < nRows; ++i) {
        std::copy(DistMatrix[i].begin(), DistMatrix[i].end(), distMatrixIn.begin() + i * nCols);
    }

    assignmentoptimal(assignment, &cost, distMatrixIn, nRows, nCols);
    Assignment = std::move(assignment);
    return cost; 
}

// Метод assignmentoptimal находит оптимальное назначение с минимальной стоимостью
void HungarianAlgorithm::assignmentoptimal(std::vector<int>& assignment, double* cost, const std::vector<double>& distMatrixIn, int nOfRows, int nOfColumns) {
    std::vector<double> distMatrix = distMatrixIn; 
    std::vector<bool> coveredColumns(nOfColumns, false), coveredRows(nOfRows, false);
    std::vector<bool> starMatrix(nOfRows * nOfColumns, false);
    int minDim = std::min(nOfRows, nOfColumns);
    *cost = 0.0;

    // Вычитаем минимальные значения из строк
    for (int i = 0; i < nOfRows; ++i) {
        double minValue = *std::min_element(distMatrix.begin() + i * nOfColumns, distMatrix.begin() + (i + 1) * nOfColumns);
        for (int j = 0; j < nOfColumns; ++j) {
            distMatrix[i * nOfColumns + j] -= minValue;
        }
    }

    // Отмечаем звезды в матрице
    for (int row = 0; row < nOfRows; ++row) {
        for (int col = 0; col < nOfColumns; ++col) {
            if (fabs(distMatrix[row * nOfColumns + col]) < std::numeric_limits<double>::epsilon() && !coveredColumns[col]) {
                starMatrix[row * nOfColumns + col] = true;
                coveredColumns[col] = true;
                break;
            }
        }
    }

    if (std::count(coveredColumns.begin(), coveredColumns.end(), true) == minDim) {
        buildassignmentvector(assignment, starMatrix, nOfRows, nOfColumns);
    } else {
        std::vector<bool> primeMatrix(nOfRows * nOfColumns, false);
        starMethod(assignment, distMatrix, starMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
    }

    computeassignmentcost(assignment, cost, distMatrixIn, nOfRows);
}

// Метод для построения вектора назначений на основе матрицы звезд
void HungarianAlgorithm::buildassignmentvector(std::vector<int>& assignment, const std::vector<bool>& starMatrix, int nOfRows, int nOfColumns) {
    for (int row = 0; row < nOfRows; ++row) {
        for (int col = 0; col < nOfColumns; ++col) {
            if (starMatrix[row * nOfColumns + col]) {
                assignment[row] = col;
                break;
            }
        }
    }
}

// Метод для вычисления стоимости назначения
void HungarianAlgorithm::computeassignmentcost(const std::vector<int>& assignment, double* cost, const std::vector<double>& distMatrix, int nOfRows) {
    *cost = std::accumulate(assignment.begin(), assignment.end(), 0.0, [&](double sum, int col) {
        return col >= 0 ? sum + distMatrix[col] : sum;
    });
}

// Метод звездного метода для нахождения оптимального назначения
void HungarianAlgorithm::starMethod(std::vector<int>& assignment, std::vector<double>& distMatrix, std::vector<bool>& starMatrix, std::vector<bool>& primeMatrix, std::vector<bool>& coveredColumns, std::vector<bool>& coveredRows, int nOfRows, int nOfColumns, int minDim) {
    bool zerosFound;
    do {
        zerosFound = false;
        for (int col = 0; col < nOfColumns; ++col) {
            if (coveredColumns[col]) continue;

            for (int row = 0; row < nOfRows; ++row) {
                if (coveredRows[row] || fabs(distMatrix[row * nOfColumns + col]) >= std::numeric_limits<double>::epsilon()) continue;

                primeMatrix[row * nOfColumns + col] = true;
                coveredRows[row] = true;

                for (int starCol = 0; starCol < nOfColumns; ++starCol) {
                    if (starMatrix[row * nOfColumns + starCol]) {
                        coveredColumns[starCol] = false;
                        zerosFound = true;
                        break;
                    }
                }

                if (!zerosFound) {
                    starMatrix[row * nOfColumns + col] = true;

                    for (int starCol = col; starCol < nOfColumns; ++starCol) {
                        for (int r = 0; r < nOfRows; ++r) {
                            if (starMatrix[r * nOfColumns + starCol]) {
                                starMatrix[r * nOfColumns + starCol] = false;
                                for (int primeCol = 0; primeCol < nOfColumns; ++primeCol) {
                                    if (primeMatrix[r * nOfColumns + primeCol]) {
                                        starMatrix[r * nOfColumns + primeCol] = true;
                                        break;
                                    }
                                }
                                break;
                            }
                        }
                    }

                    std::fill(coveredRows.begin(), coveredRows.end(), false);
                    starMethod(assignment, distMatrix, starMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, nOfRows);
                    return;
                }
            }
        }
    } while (zerosFound);

    double h = std::numeric_limits<double>::max();
    for (int row = 0; row < nOfRows; ++row) {
        if (coveredRows[row]) continue;
        for (int col = 0; col < nOfColumns; ++col) {
            if (!coveredColumns[col]) {
                h = std::min(h, distMatrix[row * nOfColumns + col]);
            }
        }
    }

    for (int row = 0; row < nOfRows; ++row) {
        if (!coveredRows[row]) {
            for (int col = 0; col < nOfColumns; ++col) {
                distMatrix[row * nOfColumns + col] += h;
            }
        }
    }

    for (int col = 0; col < nOfColumns; ++col) {
        if (!coveredColumns[col]) {
            for (int row = 0; row < nOfRows; ++row) {
                distMatrix[row * nOfColumns + col] -= h;
            }
        }
    }

    starMethod(assignment, distMatrix, starMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}
