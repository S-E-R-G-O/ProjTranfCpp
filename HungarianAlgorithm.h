#ifndef HUNGARIAN_ALGORITHM_H
#define HUNGARIAN_ALGORITHM_H

#include <iostream>
#include <vector>

using namespace std;

class HungarianAlgorithm {
public:
    HungarianAlgorithm();
    ~HungarianAlgorithm();

    double Solve(std::vector<std::vector<double>>& DistMatrix, std::vector<int>& Assignment);

private:
    void assignmentoptimal(std::vector<int>& assignment, double* cost, const std::vector<double>& distMatrixIn, int nOfRows, int nOfColumns);
    void buildassignmentvector(std::vector<int>& assignment, const std::vector<bool>& starMatrix, int nOfRows, int nOfColumns);
    void computeassignmentcost(const std::vector<int>& assignment, double* cost, const std::vector<double>& distMatrix, int nOfRows);
    void step2a(std::vector<int>& assignment, std::vector<double>& distMatrix, std::vector<bool>& starMatrix, std::vector<bool>& newStarMatrix, std::vector<bool>& primeMatrix, std::vector<bool>& coveredColumns, std::vector<bool>& coveredRows, int nOfRows, int nOfColumns, int minDim);
    void step2b(std::vector<int>& assignment, std::vector<double>& distMatrix, std::vector<bool>& starMatrix, std::vector<bool>& newStarMatrix, std::vector<bool>& primeMatrix, std::vector<bool>& coveredColumns, std::vector<bool>& coveredRows, int nOfRows, int nOfColumns, int minDim);
    void step3(std::vector<int>& assignment, std::vector<double>& distMatrix, std::vector<bool>& starMatrix, std::vector<bool>& newStarMatrix, std::vector<bool>& primeMatrix, std::vector<bool>& coveredColumns, std::vector<bool>& coveredRows, int nOfRows, int nOfColumns, int minDim);
    void step4(std::vector<int>& assignment, std::vector<double>& distMatrix, std::vector<bool>& starMatrix, std::vector<bool>& newStarMatrix, std::vector<bool>& primeMatrix, std::vector<bool>& coveredColumns, std::vector<bool>& coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col);
    void step5(std::vector<int>& assignment, std::vector<double>& distMatrix, std::vector<bool>& starMatrix, std::vector<bool>& newStarMatrix, std::vector<bool>& primeMatrix, std::vector<bool>& coveredColumns, std::vector<bool>& coveredRows, int nOfRows, int nOfColumns, int minDim);
};

#endif // HUNGARIAN_ALGORITHM_H
