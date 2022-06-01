#include <iostream>
#include <fstream>
#include "Eigen/Dense"
#include "Eigen/LU"
#include "Eigen/Sparse"

#include <math.h>
#include <vector>
#include "FEValues.h"
#include "primalEuropeanOptions.h"
#include "BinomialHighamOptionPricing.h"
#include <petscksp.h>
#include "petscmat.h"

using namespace std;
using namespace Eigen;

#define PI 3.14159265
struct shapeCom
{
    vector<double> zc;
    //VectorXd  zc;
    //VectorXd  dNdz;
    // VectorXd  d2Ndz2;
    //MatrixXd  N;


};

int main() {



    primalEuropeanOptions obj;
    //obj.run();
    //obj.BackwardEulerSolver();
    //BinomialHighamOptionPricing obj2;
    obj.findError();

    return 0;
}

