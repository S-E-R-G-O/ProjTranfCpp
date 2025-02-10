#ifndef HUNGARIAN_ALGORITHM_H
#define HUNGARIAN_ALGORITHM_H

#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>

class HungarianAlgorithm {
private:
    void assignmentoptimal(std::vector<int>& assignment, double* cost, const std::vector<double>& distMatrixIn, int nOfRows, int nOfColumns);
    void buildassignmentvector(std::vector<int>& assignment, const std::vector<bool>& starMatrix, int nOfRows, int nOfColumns);
    void computeassignmentcost(const std::vector<int>& assignment, double* cost, const std::vector<double>& distMatrix, int nOfRows);
    void starMethod(std::vector<int>& assignment, std::vector<double>& distMatrix, std::vector<bool>& starMatrix, std::vector<bool>& primeMatrix, std::vector<bool>& coveredColumns, std::vector<bool>& coveredRows, int nOfRows, int nOfColumns, int minDim);

public:
    HungarianAlgorithm() {}
    ~HungarianAlgorithm() {}

    double Solve(std::vector<std::vector<double>>& DistMatrix, std::vector<int>& Assignment);
};

#endif // HUNGARIAN_ALGORITHM_H
