#include <stdlib.h>
#include <cfloat> // для DBL_MAX
#include <cmath>  // для fabs()
#include <vector>
#include <algorithm> // для std::transform, std::min_element
#include <iostream>

#include "HungarianAlgorithm.h"


HungarianAlgorithm::HungarianAlgorithm() {}
HungarianAlgorithm::~HungarianAlgorithm() {}

double HungarianAlgorithm::Solve(std::vector<std::vector<double>>& DistMatrix, std::vector<int>& Assignment) {
    int nRows = DistMatrix.size();
    int nCols = DistMatrix[0].size();

    std::vector<double> distMatrixIn(nRows * nCols);
    std::vector<int> assignment(nRows, -1);
    double cost = 0.0;

    
    for (int i = 0; i < nRows; ++i)
        for (int j = 0; j < nCols; ++j)
            distMatrixIn[i + nRows * j] = DistMatrix[i][j];

   
    assignmentoptimal(assignment, &cost, distMatrixIn, nRows, nCols);

    Assignment = std::move(assignment); 
}

void HungarianAlgorithm::assignmentoptimal(std::vector<int>& assignment, double* cost, const std::vector<double>& distMatrixIn, int nOfRows, int nOfColumns) {
    std::vector<double> distMatrix = distMatrixIn;
    std::vector<bool> coveredColumns(nOfColumns, false);
    std::vector<bool> coveredRows(nOfRows, false);
    std::vector<bool> starMatrix(nOfRows * nOfColumns, false);
    std::vector<bool> primeMatrix(nOfRows * nOfColumns, false);
    std::vector<bool> newStarMatrix(nOfRows * nOfColumns, false);
    int minDim;

    *cost = 0.0;

    if (nOfRows <= nOfColumns) {
        minDim = nOfRows;
        // Row Reduction
        for (int row = 0; row < nOfRows; row++) {
            double minValue = *std::min_element(distMatrix.begin() + row * nOfColumns, distMatrix.begin() + (row + 1) * nOfColumns);
            std::transform(distMatrix.begin() + row * nOfColumns, distMatrix.begin() + (row + 1) * nOfColumns, distMatrix.begin() + row * nOfColumns, [minValue](double x) { return x - minValue; });
        }
    }
    else {
        minDim = nOfColumns;
        
        for (int col = 0; col < nOfColumns; col++) {
            double minValue = *std::min_element(distMatrix.begin() + nOfRows * col, distMatrix.begin() + nOfRows * (col + 1));
            std::transform(distMatrix.begin() + nOfRows * col, distMatrix.begin() + nOfRows * (col + 1), distMatrix.begin() + nOfRows * col, [minValue](double x) { return x - minValue; });
        }
    }

  
    for (int row = 0; row < nOfRows; ++row) {
        for (int col = 0; col < nOfColumns; ++col) {
            if (fabs(distMatrix[row + nOfRows * col]) < std::numeric_limits<double>::epsilon()) {
                if (!coveredColumns[col]) {
                    starMatrix[row + nOfRows * col] = true;
                    coveredColumns[col] = true;
                    break;
                }
            }
        }
    }

    // Proceed to step 2b
    step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
    computeassignmentcost(assignment, cost, distMatrixIn, nOfRows);
}

void HungarianAlgorithm::buildassignmentvector(std::vector<int>& assignment, const std::vector<bool>& starMatrix, int nOfRows, int nOfColumns) {
    for (int row = 0; row < nOfRows; ++row)
        for (int col = 0; col < nOfColumns; ++col)
            if (starMatrix[row + nOfRows * col]) {
                assignment[row] = col; // Zero-based indexing.
                break;
            }
}

void HungarianAlgorithm::computeassignmentcost(const std::vector<int>& assignment, double* cost, const std::vector<double>& distMatrix, int nOfRows) {
    for (int row = 0; row < nOfRows; row++) {
        int col = assignment[row];
        if (col >= 0)
            *cost += distMatrix[row + nOfRows * col];
    }
}

void HungarianAlgorithm::step2a(std::vector<int>& assignment, std::vector<double>& distMatrix, std::vector<bool>& starMatrix, std::vector<bool>& newStarMatrix, std::vector<bool>& primeMatrix, std::vector<bool>& coveredColumns, std::vector<bool>& coveredRows, int nOfRows, int nOfColumns, int minDim) {
    for (int col = 0; col < nOfColumns; col++) {
        for (int row = 0; row < nOfRows; row++)
            if (starMatrix[row + nOfRows * col]) {
                coveredColumns[col] = true;
                break;
            }
    }
    step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

void HungarianAlgorithm::step2b(std::vector<int>& assignment, std::vector<double>& distMatrix, std::vector<bool>& starMatrix, std::vector<bool>& newStarMatrix, std::vector<bool>& primeMatrix, std::vector<bool>& coveredColumns, std::vector<bool>& coveredRows, int nOfRows, int nOfColumns, int minDim) {
    int nOfCoveredColumns = std::count(coveredColumns.begin(), coveredColumns.end(), true);
    
    if (nOfCoveredColumns == minDim) {
        buildassignmentvector(assignment, starMatrix, nOfRows, nOfColumns);
    } else {
        step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
    }
}

void HungarianAlgorithm::step3(std::vector<int>& assignment, std::vector<double>& distMatrix, std::vector<bool>& starMatrix, std::vector<bool>& newStarMatrix, std::vector<bool>& primeMatrix, std::vector<bool>& coveredColumns, std::vector<bool>& coveredRows, int nOfRows, int nOfColumns, int minDim) {
    bool zerosFound = true;
    while (zerosFound) {
        zerosFound = false;
        for (int col = 0; col < nOfColumns; col++)
            if (!coveredColumns[col])
                for (int row = 0; row < nOfRows; row++)
                    if (!coveredRows[row] && (fabs(distMatrix[row + nOfRows * col]) < std::numeric_limits<double>::epsilon())) {
                        primeMatrix[row + nOfRows * col] = true;
                        for (int starCol = 0; starCol < nOfColumns; starCol++)
                            if (starMatrix[row + nOfRows * starCol]) {
                                coveredRows[row] = true;
                                coveredColumns[starCol] = false;
                                zerosFound = true;
                                break;
                            }
                        
                        if (!zerosFound) {
                            step4(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim, row, col);
                            return;
                        }
                    }
    }
    step5(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

void HungarianAlgorithm::step4(std::vector<int>& assignment, std::vector<double>& distMatrix, std::vector<bool>& starMatrix, std::vector<bool>& newStarMatrix, std::vector<bool>& primeMatrix, std::vector<bool>& coveredColumns, std::vector<bool>& coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col) {
    newStarMatrix = starMatrix;
    newStarMatrix[row + nOfRows * col] = true;
    
    for (int starCol = col; starCol < nOfColumns; ++starCol) {
        int starRow = -1;
        for (int r = 0; r < nOfRows; ++r) {
            if (starMatrix[r + nOfRows * starCol]) {
                starRow = r;
                break;
            }
        }
        if (starRow == -1) break;

        newStarMatrix[starRow + nOfRows * starCol] = false;
        for (int primeCol = 0; primeCol < nOfColumns; primeCol++) {
            if (primeMatrix[starRow + nOfRows * primeCol]) {
                newStarMatrix[starRow + nOfRows * primeCol] = true;
                break;
            }
        }
    }

    starMatrix = std::move(newStarMatrix);
    std::fill(coveredRows.begin(), coveredRows.end(), false);
    step2a(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

void HungarianAlgorithm::step5(std::vector<int>& assignment, std::vector<double>& distMatrix, std::vector<bool>& starMatrix, std::vector<bool>& newStarMatrix, std::vector<bool>& primeMatrix, std::vector<bool>& coveredColumns, std::vector<bool>& coveredRows, int nOfRows, int nOfColumns, int minDim) {
    double h = std::numeric_limits<double>::max();

    for (int row = 0; row < nOfRows; row++)
        if (!coveredRows[row])
            for (int col = 0; col < nOfColumns; col++)
                if (!coveredColumns[col])
                    h = std::min(h, distMatrix[row + nOfRows * col]);

    for (int row = 0; row < nOfRows; row++)
        if (coveredRows[row])
            for (int col = 0; col < nOfColumns; col++)
                distMatrix[row + nOfRows * col] += h;

    for (int col = 0; col < nOfColumns; col++)
        if (!coveredColumns[col])
            for (int row = 0; row < nOfRows; row++)
                distMatrix[row + nOfRows * col] -= h;

    step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}
