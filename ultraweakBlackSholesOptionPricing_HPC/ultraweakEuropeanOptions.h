//
// Created by Davood Damirchelli on 2021-12-09.
//
#include <iostream>
#include <fstream>
#include "Eigen/Dense"
#include "Eigen/LU"
#include <math.h>
#include <string>
#include <vector>
#include "FEValues.h"
#include <petscksp.h>
#include "petscmat.h"
#include "Eigen/Sparse"

#ifndef ULTRAWEAKDPGOPTIONPRICING_ULTRAWEAKEUROPEANOPTIONS_H
#define ULTRAWEAKDPGOPTIONPRICING_ULTRAWEAKEUROPEANOPTIONS_H

using namespace std;
using namespace Eigen;


class ultraweakEuropeanOptions {


public:

    VectorXd generalError;
    double errorL2;
    double errorLinf;

    ///----Constructor
    ultraweakEuropeanOptions();
    ///-----Methods-----
    void run();
    void BackwardEulerSolver();
    void findError();

private:
    ///---------Private Methods
    void setInitialValue();
    void findInitialValue();
    void decomposeSol();
    void assembly();
    void lgwt(int dim, double I0_=-1.0, double I1_=1.0);
    void optionValue();
    void imposeFreeBoundary();
    void saveTotalSolution();
    void saveResults();
    double Euro9PutDesHigham(float x);
    VectorXd AmericanPutDesHigham(int x);
    void postProcessingTrial();
    MatrixXd  myTranspose(VectorXd vec); // this is method JUST for vectors
    void PETScVectoEigenVec(Vec& pVec, int n, VectorXd& eVec);
    ///--------- Main blocks of DPG
    MatrixXd G_ ;
    VectorXd F_ ;
    MatrixXd B_ ;
    ///------------DPG Parameters-------------

    int   Nelems_    = 200;
    int      Nt_     = 10;  ///This must be greater than T_
    int   p_         = 2;  //for p>1 change FEValues

    ///------------Used classes
    FEValues<2> feTrail;   // T=p_
    FEValues<4> feTest;


    int   trilDim_   = p_;
    int   testDim_   = p_+ 2;
    int   tracDim_   = 1;
    int   quadPoints = 2*p_ + 4;
    float teta_      = 1.0;   // still is not working
    float beta_      = 0.5;   // test space norm parameter
    int NU_ ;                 //trial Dof
    int NV_ ;                 //test Dof
    int Nuh_;                 //trace Dof
    int NuQh_;                // flux Dof
    int locDof_;
    int base_;
    //int shift_;
    int n_;
    int m_;
    ///------------ Model Parameters-------------
    int optionType_   = 1;     //0:European, 1:American
    int contractType_ = 1;     //0: call, 1:put
    float sigma_      = 0.15;
    float r_          = 0.05;
    int   k_          = 100;
    ///----------- Mesh parameters-------------
    float    A0_ = -6.0;
    float    A1_ =  6.0;
    int      Nnodes_   ;
    float    length_   ;
    double   h_        ;
    float    dxdz_     ;
    VectorXd x_        ;
    ///------------ Time discrete parameters---
    float    T_   = 1.0;
    //int      Nt_  = 10;  ///This must be greater thatn T_
    float    dt_  = T_/(Nt_-1);

    ///-------------Auxilary parameters -------
    float cof1_;
    float cof2_;
    float cof3_;
    int  Ntdof_;    // total degrees of freedom
    ///-----------Auxilary blocks ------------

    VectorXd Uinital_;
    VectorXd Ulocinital_;   // this is just for first trial variable
    VectorXd Utotal_;
    VectorXd Utrial_;
    //VectorXd Utrac_;
    //VectorXd Uflux_;
    VectorXd uVec_;
    VectorXd sigVec_;
    VectorXd uHat_;
    VectorXd qnHat_;
    VectorXd qpLoc;
    VectorXd qpWgt;

    ///------------PETSc parameters
    KSP            ksp_;          /* linear solver context */
    PC             pc_;           /* preconditioner context */
    PetscMPIInt    rank_;
    MPI_Comm       comm_;
    Mat Bpetsc_;
    Mat Gpetsc_;
    Vec Fpetsc_;
    Vec Sol_;
    int argc;
    char **args;

};


#endif //ULTRAWEAKDPGOPTIONPRICING_ULTRAWEAKEUROPEANOPTIONS_H
