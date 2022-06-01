#include <iostream>
#include <fstream>
#include "Eigen/Dense"
#include "Eigen/LU"
#include "Eigen/Sparse"
#include <math.h>
#include <vector>
#include "FEValues.h"
#include <petscksp.h>
#include "petscmat.h"

#include "ultraweakEuropeanOptions.h"

using namespace std;
using namespace Eigen;

#define PI 3.14159265


int main()
{

    ultraweakEuropeanOptions obj;
    //obj.run();
    obj.findError();
    std::cout << "I am writing ultraweak for option pricing" << std::endl;
    return 0;

}