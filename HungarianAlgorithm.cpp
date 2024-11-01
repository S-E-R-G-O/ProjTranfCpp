

#include <stdlib.h>
#include <cfloat> // для DBL_MAX
#include <cmath>  // для fabs()
#include "HungarianAlgorithm.h"


HungarianAlgorithm::HungarianAlgorithm() {}
HungarianAlgorithm::~HungarianAlgorithm() {}


double HungarianAlgorithm::Solve(vector <vector<double> >& DistMatrix, vector<int>& Assignment)
{
    unsigned int nRows = DistMatrix.size();
    unsigned int nCols = DistMatrix[0].size();

    double* distMatrixIn = new double[nRows * nCols];
    int* assignment = new int[nRows];
    double cost = 0.0;

    
    for (unsigned int i = 0; i < nRows; i++)
        for (unsigned int j = 0; j < nCols; j++)
            distMatrixIn[i + nRows * j] = DistMatrix[i][j];

    // вызов функции решения
    assignmentoptimal(assignment, &cost, distMatrixIn, nRows, nCols);

    Assignment.clear();
    for (unsigned int r = 0; r < nRows; r++)
        Assignment.push_back(assignment[r]);

    delete[] distMatrixIn;
    delete[] assignment;
    return cost;
}


//********************************************************//
// Решение для оптимального решения задачи о назначении с использованием 
// алгоритма Мункреса, также известного как Венгерский алгоритм.
//********************************************************//
void HungarianAlgorithm::assignmentoptimal(int* assignment, double* cost, double* distMatrixIn, int nOfRows, int nOfColumns)
{
    double* distMatrix, * distMatrixTemp, * distMatrixEnd, * columnEnd, value, minValue;
    bool* coveredColumns, * coveredRows, * starMatrix, * newStarMatrix, * primeMatrix;
    int nOfElements, minDim, row, col;

    /* инициализация */
    *cost = 0;
    for (row = 0; row < nOfRows; row++)
        assignment[row] = -1;

    /* создание рабочей копии матрицы расстояний */
    /* проверка на все элементы матрицы - положительные */
    nOfElements = nOfRows * nOfColumns;
    distMatrix = (double*)malloc(nOfElements * sizeof(double));
    distMatrixEnd = distMatrix + nOfElements;

    for (row = 0; row < nOfElements; row++)
    {
        value = distMatrixIn[row];
        if (value < 0)
            cerr << "Все элементы матрицы должны быть неотрицательными." << endl;
        distMatrix[row] = value;
    }

    /* выделение памяти */
    coveredColumns = (bool*)calloc(nOfColumns, sizeof(bool));
    coveredRows = (bool*)calloc(nOfRows, sizeof(bool));
    starMatrix = (bool*)calloc(nOfElements, sizeof(bool));
    primeMatrix = (bool*)calloc(nOfElements, sizeof(bool));
    newStarMatrix = (bool*)calloc(nOfElements, sizeof(bool)); /* используется на шаге 4 */

    /* предварительные шаги */
    if (nOfRows <= nOfColumns)
    {
        minDim = nOfRows;

        for (row = 0; row < nOfRows; row++)
        {
            /* найти наименьший элемент в строке */
            distMatrixTemp = distMatrix + row;
            minValue = *distMatrixTemp;
            distMatrixTemp += nOfRows;
            while (distMatrixTemp < distMatrixEnd)
            {
                value = *distMatrixTemp;
                if (value < minValue)
                    minValue = value;
                distMatrixTemp += nOfRows;
            }

            /* вычесть наименьший элемент из каждого элемента строки */
            distMatrixTemp = distMatrix + row;
            while (distMatrixTemp < distMatrixEnd)
            {
                *distMatrixTemp -= minValue;
                distMatrixTemp += nOfRows;
            }
        }

        /* Шаги 1 и 2а */
        for (row = 0; row < nOfRows; row++)
            for (col = 0; col < nOfColumns; col++)
                if (fabs(distMatrix[row + nOfRows * col]) < DBL_EPSILON)
                    if (!coveredColumns[col])
                    {
                        starMatrix[row + nOfRows * col] = true;
                        coveredColumns[col] = true;
                        break;
                    }
    }
    else /* if(nOfRows > nOfColumns) */
    {
        minDim = nOfColumns;

        for (col = 0; col < nOfColumns; col++)
        {
            /* найти наименьший элемент в столбце */
            distMatrixTemp = distMatrix + nOfRows * col;
            columnEnd = distMatrixTemp + nOfRows;

            minValue = *distMatrixTemp++;
            while (distMatrixTemp < columnEnd)
            {
                value = *distMatrixTemp++;
                if (value < minValue)
                    minValue = value;
            }

            /* вычесть наименьший элемент из каждого элемента столбца */
            distMatrixTemp = distMatrix + nOfRows * col;
            while (distMatrixTemp < columnEnd)
                *distMatrixTemp++ -= minValue;
        }

        /* Шаги 1 и 2а */
        for (col = 0; col < nOfColumns; col++)
            for (row = 0; row < nOfRows; row++)
                if (fabs(distMatrix[row + nOfRows * col]) < DBL_EPSILON)
                    if (!coveredRows[row])
                    {
                        starMatrix[row + nOfRows * col] = true;
                        coveredColumns[col] = true;
                        coveredRows[row] = true;
                        break;
                    }
        for (row = 0; row < nOfRows; row++)
            coveredRows[row] = false;
    }

    /* переход к шагу 2b */
    step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);

    /* вычисление стоимости и удаление недействительных назначений */
    computeassignmentcost(assignment, cost, distMatrixIn, nOfRows);

    /* освобождение выделенной памяти */
    free(distMatrix);
    free(coveredColumns);
    free(coveredRows);
    free(starMatrix);
    free(primeMatrix);
    free(newStarMatrix);

    return;
}

/********************************************************/
void HungarianAlgorithm::buildassignmentvector(int* assignment, bool* starMatrix, int nOfRows, int nOfColumns)
{
    int row, col;

    for (row = 0; row < nOfRows; row++)
        for (col = 0; col < nOfColumns; col++)
            if (starMatrix[row + nOfRows * col])
            {
#ifdef ONE_INDEXING
                assignment[row] = col + 1; /* Индексация MATLAB */
#else
                assignment[row] = col;
#endif
                break;
            }
}

/********************************************************/
void HungarianAlgorithm::computeassignmentcost(int* assignment, double* cost, double* distMatrix, int nOfRows)
{
    int row, col;

    for (row = 0; row < nOfRows; row++)
    {
        col = assignment[row];
        if (col >= 0)
            *cost += distMatrix[row + nOfRows * col];
    }
}

/********************************************************/
void HungarianAlgorithm::step2a(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix, bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim)
{
    bool* starMatrixTemp, * columnEnd;
    int col;

    /* покрыть каждый столбец, содержащий звезду ноль */
    for (col = 0; col < nOfColumns; col++)
    {
        starMatrixTemp = starMatrix + nOfRows * col;
        columnEnd = starMatrixTemp + nOfRows;
        while (starMatrixTemp < columnEnd) {
            if (*starMatrixTemp++)
            {
                coveredColumns[col] = true;
                break;
            }
        }
    }

    /* переход к шагу 3 */
    step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

/********************************************************/
void HungarianAlgorithm::step2b(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix, bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim)
{
    int col, nOfCoveredColumns;

    /* подсчитать количество покрытых столбцов */
    nOfCoveredColumns = 0;
    for (col = 0; col < nOfColumns; col++)
        if (coveredColumns[col])
            nOfCoveredColumns++;

    if (nOfCoveredColumns == minDim)
    {
        /* алгоритм завершен */
        buildassignmentvector(assignment, starMatrix, nOfRows, nOfColumns);
    }
    else
    {
        /* переход к шагу 3 */
        step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
    }
}

/********************************************************/
void HungarianAlgorithm::step3(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix, bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim)
{
    bool zerosFound;
    int row, col, starCol;

    zerosFound = true;
    while (zerosFound)
    {
        zerosFound = false;
        for (col = 0; col < nOfColumns; col++)
            if (!coveredColumns[col])
                for (row = 0; row < nOfRows; row++)
                    if ((!coveredRows[row]) && (fabs(distMatrix[row + nOfRows * col]) < DBL_EPSILON))
                    {
                        /* пометить ноль */
                        primeMatrix[row + nOfRows * col] = true;

                        /* найти звезду ноль в текущей строке */
                        for (starCol = 0; starCol < nOfColumns; starCol++)
                            if (starMatrix[row + nOfRows * starCol])
                                break;

                        if (starCol == nOfColumns) /* звезда ноль не найдена */
                        {
                            /* переход к шагу 4 */
                            step4(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim, row, col);
                            return;
                        }
                        else
                        {
                            coveredRows[row] = true;
                            coveredColumns[starCol] = false;
                            zerosFound = true;
                            break;
                        }
                    }
    }

    /* переход к шагу 5 */
    step5(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

/********************************************************/
void HungarianAlgorithm::step4(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix, bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col)
{
    int n, starRow, starCol, primeRow, primeCol;
    int nOfElements = nOfRows * nOfColumns;

    /* создать временную копию starMatrix */
    for (n = 0; n < nOfElements; n++)
        newStarMatrix[n] = starMatrix[n];

    /* пометить текущий ноль */
    newStarMatrix[row + nOfRows * col] = true;

    /* найти звезду ноль в текущем столбце */
    starCol = col;
    for (starRow = 0; starRow < nOfRows; starRow++)
        if (starMatrix[starRow + nOfRows * starCol])
            break;

    while (starRow < nOfRows)
    {
        /* снять знак с отмеченного нуля */
        newStarMatrix[starRow + nOfRows * starCol] = false;

        /* найти помеченный ноль в текущей строке */
        primeRow = starRow;
        for (primeCol = 0; primeCol < nOfColumns; primeCol++)
            if (primeMatrix[primeRow + nOfRows * primeCol])
                break;

        /* пометить помеченный ноль */
        newStarMatrix[primeRow + nOfRows * primeCol] = true;

        /* найти звезду ноль в текущем столбце */
        starCol = primeCol;
        for (starRow = 0; starRow < nOfRows; starRow++)
            if (starMatrix[starRow + nOfRows * starCol])
                break;
    }

    /* использовать временную копию как новую starMatrix */
    /* удалить все пометки, раскрыть все строки */
    for (n = 0; n < nOfElements; n++)
    {
        primeMatrix[n] = false;
        starMatrix[n] = newStarMatrix[n];
    }
    for (n = 0; n < nOfRows; n++)
        coveredRows[n] = false;

    /* переход к шагу 2а */
    step2a(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

/********************************************************/
void HungarianAlgorithm::step5(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix, bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim)
{
    double h, value;
    int row, col;

    /* найти наименьший незакрытый элемент h */
    h = DBL_MAX;
    for (row = 0; row < nOfRows; row++)
        if (!coveredRows[row])
            for (col = 0; col < nOfColumns; col++)
                if (!coveredColumns[col])
                {
                    value = distMatrix[row + nOfRows * col];
                    if (value < h)
                        h = value;
                }

    /* добавить h к каждому закрытому ряду */
    for (row = 0; row < nOfRows; row++)
        if (coveredRows[row])
            for (col = 0; col < nOfColumns; col++)
                distMatrix[row + nOfRows * col] += h;

    /* вычесть h из каждого незакрытого столбца */
    for (col = 0; col < nOfColumns; col++)
        if (!coveredColumns[col])
            for (row = 0; row < nOfRows; row++)
                distMatrix[row + nOfRows * col] -= h;

    /* переход к шагу 3 */
    step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}
