//
// Created by Davood Damirchelli on 2021-10-19.
//
#include <iostream>
#include <fstream>
#include "Eigen/Dense"
#include "Eigen/LU"
#include <math.h>
#include <string>
#include <vector>
#include "Eigen/Sparse"
#include <petscksp.h>
#include "petscmat.h"

#include "primalEuropeanOptions.h"

static char help[] = "Solves a tridiagonal linear system with KSP.\n\n";

using namespace std;
using namespace Eigen;
#define PI 3.14159265
primalEuropeanOptions::primalEuropeanOptions()
{

    Nnodes_ = trilDim_ * Nelems_ + 1 ;
    length_ = A1_-A0_   ;
    h_      = length_ /Nelems_;
    x_      = VectorXd::LinSpaced(Nnodes_,A0_,A1_);
    dxdz_   = h_/2.0;
    cof1_   = 0.5 * pow(sigma_,2);
    cof2_   = r_ - 0.5 * pow(sigma_,2);
    cof3_   = r_;
    //Ntdof_  = 2.0 * Nnodes_;    //Total degree of freedom //This is right just for PRIMAL!!!!

    NU_     = trilDim_ + 1 ; //trial Dof
    NV_     = testDim_ + 1;  //test Dof
    Nuh_    = tracDim_;      //trace Dof

    n_      =  NV_*Nelems_;
    m_      = (NU_-1)*Nelems_ + 1 + Nuh_ * Nelems_+1;
    Ntdof_  = m_;


    PetscInitialize(&argc,&args,(char*)0,help);
    comm_ = PETSC_COMM_WORLD;
    MPI_Comm_rank(comm_,&rank_);

    //MPI_Comm_size(PETSC_COMM_WORLD,&size_);
}

void primalEuropeanOptions::run()
{
    cout<<Nnodes_<<endl;

    //setInitialValue();
    //decomposeSol();
    //lgwt(3,-1,1);
    //assembly();
    //BackwardEulerSolver();
    //AmericanPutDesHigham(10);
    //savePrimalSolution();

}

///-------------- Implementation of Private Method---------

void primalEuropeanOptions::PETScVectoEigenVec(Vec& pVec, int n, VectorXd& eVec)
{
    double vali_ ;
    for(int row=0; row<n;row++)
    {
        VecGetValues(pVec,1,&row, &vali_);
        eVec(row) = vali_;
    }

}
MatrixXd  primalEuropeanOptions::myTranspose(VectorXd vec)
{
    int sizeVec_  = vec.size();

    MatrixXd vecT_(1,sizeVec_ );

    for(int i=0; i<sizeVec_;i++)
    {
        vecT_(0,i) = vec(i);
    }

    return vecT_;
}
void primalEuropeanOptions::setInitialValue()
{
    VectorXd Xsol;
    Xsol = VectorXd::Zero(Ntdof_);
    float ax1_;

      switch (contractType_)
      {
          case 0:          //0: call,
          {
              for(int i=0;i<=Nnodes_-1;i++)
              {
                  ax1_ = exp(x_(i)) - k_;

                  Xsol(i) = max(ax1_, 0.0f);
              }

              Uinital_ = Xsol;
              break;
          }

          case 1:         // 1:put
          {
              for(int i=0;i<=Nnodes_-1;i++)
              {
                  ax1_ = k_ - exp(x_(i));
                  Xsol(i) = max(ax1_, 0.0f);
              }
              Uinital_ = Xsol;
              break;
          }
      }
}

void primalEuropeanOptions::lgwt(int dim_, double a, double b)
    {
        /*
         *  This script is for computing definite integrals using Legendre-Gauss
             Quadrature. Computes the Legendre-Gauss nodes and weights  on an interval
             [a,b] with truncation order N

             Suppose you have a continuous function f(x) which is defined on [a,b]
             which you can evaluate at any x in [a,b]. Simply evaluate it at all of
             the values contained in the x vector to obtain a vector f. Then compute
             the definite integral using sum(f.*w);
         *
         */

        dim_  = dim_-1;

        double N1 = dim_+1;
        double N2 = dim_+2;
        VectorXd xu = VectorXd::LinSpaced(N1,-1,1);
        VectorXd p1;
        VectorXd p2;

        VectorXd p3=VectorXd::Zero(dim_+1);
        VectorXd one; one.setOnes(dim_+1,1);
        VectorXd ax;
        MatrixXd L;
        MatrixXd Lp;
        VectorXd y0;
        VectorXd x = VectorXd::Zero(dim_+1);
        VectorXd ax2;

        VectorXd w = VectorXd::Zero(dim_+1);

        qpLoc = VectorXd::Zero(dim_+1);
        qpWgt = VectorXd::Zero(dim_+1);

        p1 = VectorXd::LinSpaced(dim_+1,0,dim_);      // low:step:hi
        p1 = ((2* p1+one)*PI)/(2*dim_+2);
        p2 = (PI*dim_/N2)*xu;

        //Initial guess
        p3 = cos(p1.array())+(0.27/N1)*(sin(p2.array()));
        // Legendre-Gauss Vandermonde Matrix
        L  = MatrixXd::Zero(N1,N2);
        y0 = 2*y0.setOnes(dim_+1,1);
        // Compute the zeros of the N+1 Legendre Polynomial
        // using the recursion relation and the Newton-Raphson method
        double eps = 0.0000000000001;
        int cont = 0;

        while (((p3-y0).array().abs()).maxCoeff()>eps && cont<15)
        {

            L.col(0)  = one;
            L.col(1)  = p3;

            for (int k = 2; k <= N1; k++)
            {
                L.col(k) = (1.0/k)*(((2*k-1)*p3).cwiseProduct(L.col(k-1)).array()-(k-1)*L.col(k-2).array());
            }

            ax = one.array() - (p3.array().pow(2)).array();
            Lp=(N2)* (L.col(N1-1)-p3.cwiseProduct(L.col(N2-1))).cwiseQuotient(ax);
            y0 = p3;
            p3 = y0-(L.col(N2-1)).cwiseQuotient(Lp);
            cont++;
        }

        //Linear map from[-1,1] to [a,b]

        x=(a*(one-p3)+b*(one+p3))/2.0;

        //Compute the weights

        ax2 = ((1.0-p3.array().pow(2)).cwiseProduct(Lp.array().pow(2)));

        w = ((b-a)*pow(N2/N1,2))*(ax2).cwiseInverse();

        qpLoc = x;
        qpWgt = w;

    }
void primalEuropeanOptions::decomposeSol()
{
    int a_ = Nnodes_;
    int b_ = Ntdof_;
    Utotal_= Uinital_;
    Utrial_= Utotal_.segment(0, a_);
    Utrac_ = Utotal_.segment(a_-1, b_-a_);

///?????
    //cout<<"I need to check this decomposeSol()"<<Utrac_<<endl;
}
void primalEuropeanOptions::assembly()
{
    ///-------- PETSc init-----------
        //preparing some stuff for petsc
    int indU_[NU_];
    int indV_[NV_];
    int indUhat_[Nuh_+1];
    ///-----------PETSc Objects
    // creating PETSc Matirces
    MatCreate(comm_,&Gpetsc_);
    MatSetSizes(Gpetsc_,PETSC_DECIDE,PETSC_DECIDE,n_,n_);
    MatSetFromOptions(Gpetsc_);
    MatSetType(Gpetsc_,MATMPIAIJ);
    // MatMPIAIJSetPreallocation(Gpetsc_, NV_, NULL,0, NULL);
    MatSetUp(Gpetsc_);

    MatCreate(PETSC_COMM_WORLD,&Bpetsc_);
    MatSetSizes(Bpetsc_,PETSC_DECIDE,PETSC_DECIDE,n_,m_);
    MatSetFromOptions(Bpetsc_);
    MatSetType(Bpetsc_,MATMPIAIJ);
    // MatMPIAIJSetPreallocation(Bpetsc_,NV_,NULL,2,NULL);
    MatSetUp(Bpetsc_);
    // creating vectors

    VecCreate(PETSC_COMM_WORLD,&Fpetsc_);
    VecSetType(Fpetsc_,VECMPI);
    PetscObjectSetName((PetscObject) Fpetsc_, "RHS");
    VecSetSizes(Fpetsc_,PETSC_DECIDE,n_);
    VecSetFromOptions(Fpetsc_);

    MatrixXd tempGe_    = MatrixXd::Zero(NV_,NV_);
    MatrixXd tempBe_    = MatrixXd::Zero(NV_,NU_);
    MatrixXd tempBhate_ = MatrixXd::Zero(NV_,Nuh_+1);

    MatrixXd tempGeT_    = MatrixXd::Zero(NV_,NV_);
    MatrixXd tempBeT_    = MatrixXd::Zero(NU_,NV_);
    MatrixXd tempBhateT_ = MatrixXd::Zero(Nuh_+1,NV_);

    VectorXd tempFe_     = VectorXd::Zero(NV_);

////???????????????

    MatZeroEntries(Gpetsc_);
    MatZeroEntries(Bpetsc_);
    VecZeroEntries(Fpetsc_);




    ///-------- PETSc init-----------END


    MatrixXd axB_(NV_*Nelems_,(NU_-1)*Nelems_ + 1 + Nuh_*Nelems_+1);
    MatrixXd axG_(NV_*Nelems_,NV_*Nelems_);
    VectorXd axF_(NV_*Nelems_);



    MatrixXd A1mat_;
    MatrixXd A2mat_;
    MatrixXd A3mat_;
    MatrixXd A4mat_;
    MatrixXd d1mat_(NV_,Nuh_+1);
    MatrixXd Ge_(NV_,NV_);
    VectorXd Fe1_(NV_);
    VectorXd Fe2_(NV_);
    VectorXd Fe3_(NV_);
    VectorXd Fe4_(NV_);
    VectorXd Fe5_(NV_);
    VectorXd idxU_;
    VectorXd idxV_;
    VectorXd idxUh_;
    VectorXd N_;
    VectorXd dN_dz_;
    VectorXd M_;
    VectorXd dM_dz_;



    int a1; int a2;int b1;int b2;int a3;int b3;int baseIdx;int baseUhat;
    baseUhat = (NU_- 1) * Nelems_ + 1 - 1;
    ///---------- initializing------

    axB_ = MatrixXd::Zero(NV_*Nelems_,(NU_-1)*Nelems_ + 1 + Nuh_*Nelems_+1);
    axG_ = MatrixXd::Zero(NV_*Nelems_,NV_*Nelems_);

    axF_ = VectorXd::Zero(NV_*Nelems_);
    ///---------- quad values-----

    lgwt(quadPoints,-1,1);

    decomposeSol();

for(int i=1; i<=Nelems_;i++)
{
    A1mat_ = MatrixXd::Zero(NV_,NU_);
    A2mat_ = MatrixXd::Zero(NV_,NU_);
    A3mat_ = MatrixXd::Zero(NV_,NU_);
    A4mat_ = MatrixXd::Zero(NV_,NU_);

    d1mat_ = MatrixXd::Zero(NV_,Nuh_+1); d1mat_(0,0) = -1.0;d1mat_(NV_-1,Nuh_) = 1.0;
    Ge_    = MatrixXd::Zero(NV_,NV_);

    Fe1_   = VectorXd::Zero(NV_);

    Fe2_   = VectorXd::Zero(NV_);
    Fe3_   = VectorXd::Zero(NV_);
    Fe4_   = VectorXd::Zero(NV_);
    Fe5_   = VectorXd::Zero(NV_);

    a1  = (NU_-1) * (i) - ((NU_-1)-1)-1;
    b1  = (NU_-1) * (i) + 1-1;
    a2  = NV_ * (i-1) + 1-1;
    b2  = NV_ * (i)-1;

    baseIdx =  Nelems_ * (NU_-1)+ 1-1;
    a3 = baseIdx + Nuh_ * i - (Nuh_-1)-1;

    b3 = baseIdx + Nuh_ * i + 1 -1;

    //VectorXd::LinSpaced(size,low,high)
    idxU_  = VectorXd::LinSpaced(b1-a1+1,a1,b1);
    idxV_  = VectorXd::LinSpaced(b2-a2+1,a2,b2);
    idxUh_ = VectorXd::LinSpaced(b3-a3+1,a3,b3);

    MatrixXd Ae4_;
    MatrixXd axFe4_;
    for(int j = 0;j<=quadPoints-1;j++)
    {
        N_     = feTrail.shapVal(qpLoc(j));
        dN_dz_ = feTrail.shapDVal(qpLoc(j)) * 1.0/dxdz_;
        M_     = feTest.shapVal(qpLoc(j));
        dM_dz_ = feTest.shapDVal(qpLoc(j))  * 1.0/dxdz_;

        ///--------Forming Bilinear form (Primal)---------
        //(u',c1v')_Ii
        A1mat_ +=  qpWgt(j) * cof1_ * dM_dz_ * myTranspose(dN_dz_) * dxdz_ ;

        //(c2u',v)_Ii


        A2mat_ +=  qpWgt(j) * cof2_ * M_ * myTranspose(dN_dz_)  * dxdz_ ;

        //(u,rv)_Ii
        A3mat_ +=  qpWgt(j) * cof3_ * M_ * myTranspose(N_)  * dxdz_ ;

        //(u,v)  ~ (u^{n+1},v)
        Ae4_ = M_ * myTranspose(N_);
        A4mat_ += qpWgt(j) *  Ae4_ * dxdz_ ;


        ///----------- Forming Gram Matrix---------------
        ///(v',v') (1.0/dt_) *(1.0/dt_)
        Ge_ += qpWgt(j) * (dt_) * dM_dz_ * myTranspose(dM_dz_)  * dxdz_;

        ///(v ,v) /dt_
        Ge_ += qpWgt(j) * (1.0 + beta_) * M_ * myTranspose(M_)  * dxdz_;

        ///----------Forming RHS -------------------------
        //Utrial_.segment(a1,b1-a1+1).transpose();

        //Utrac_.segment(a3-baseUhat,b3-a3+1).transpose();

        Fe1_ +=  qpWgt(j) *  M_ * (Utrial_.segment(a1,b1-a1+1).transpose() * N_)  * dxdz_;


        ///------------Theta Coefficients---------------
        // (c1u',v')
        Fe2_ +=  qpWgt(j) * cof1_ * dM_dz_ * (Utrial_.segment(a1,b1-a1+1).transpose() * dN_dz_)  * dxdz_;
        // (c2u',v)
        Fe3_ +=  qpWgt(j) * cof2_ * M_ * (Utrial_.segment(a1,b1-a1+1).transpose() * dN_dz_)  * dxdz_;
        // (ru,v)
        Fe4_ +=  qpWgt(j) * cof3_ *  M_ *  (Utrial_.segment(a1,b1-a1+1).transpose() * N_)  * dxdz_;

    }

    ///--------------Petsc objects--------

    // Preparing Indexes for PETSc
    for(int iu = 0;iu<=NU_-1;iu++)
    {
        indU_[iu] = a1 + iu ;
    }
    for(int iv = 0;iv<=NV_-1;iv++)
    {
        indV_[iv] = a2 + iv ;
    }
    for(int iuh = 0;iuh<=Nuh_;iuh++)///????????????
    {
        indUhat_[iuh] = a3 + iuh + 1 ;
    }

    tempGe_    = Ge_.inverse();
    tempBe_    = A4mat_-teta_*dt_*(-A1mat_+A2mat_-A3mat_);
    tempBhate_ = -teta_*dt_* d1mat_;
    tempFe_    = Fe1_;//+(1.0-teta_)*dt_*(-Fe2_+Fe3_-Fe4_);  //teta method!!!!

    tempGeT_   = tempGe_.transpose();
    tempBeT_   = tempBe_.transpose();
    tempBhateT_= tempBhate_.transpose();
    double *dataF = tempFe_.data();

    ////-------??????????for ultraweak formualtion pay attention to INSERT_VALUES or add_values!!

    MatSetValues(Gpetsc_,NV_,indV_,NV_,indV_,tempGeT_.data(),INSERT_VALUES);
    MatSetValues(Bpetsc_,NV_,indV_,NU_,indU_,tempBeT_.data(),INSERT_VALUES);
    MatSetValues(Bpetsc_,NV_,indV_,Nuh_+1,indUhat_,tempBhateT_.data(),INSERT_VALUES);
    //VecSetValues(Fpetsc_,NV_,indV_,tempFe_.data(),INSERT_VALUES);
    for(int i=0;i<NV_;i++)
    {
        VecSetValue(Fpetsc_,indV_[i],dataF[i],INSERT_VALUES);
    }

}

    MatAssemblyBegin(Gpetsc_,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Gpetsc_,MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(Bpetsc_,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Bpetsc_,MAT_FINAL_ASSEMBLY);
    VecAssemblyBegin(Fpetsc_);
    VecAssemblyEnd(Fpetsc_);

    //cout<<"this is B"<<endl;
    //VecView(Fpetsc_,0);
    //cout<<"this is B"<<endl;
     //MatView(Bpetsc_,0);

    //MatView(Gpetsc_,0);
}
void primalEuropeanOptions::optionValue()
{

    Mat AnearOptiPETSc_ ;
    Vec lnearOptiPETSc_ ;
    Mat BTGpetsc_;


    VecCreate(PETSC_COMM_WORLD,&Sol_);
    PetscObjectSetName((PetscObject) Sol_, "RHS");
    VecSetSizes(Sol_,PETSC_DECIDE,m_);
    VecSetFromOptions(Sol_);

    VecDuplicate(Sol_,&lnearOptiPETSc_);

    ///---------------  PETSc ------------------

    // C=A^T*B.
    MatTransposeMatMult(Bpetsc_,Gpetsc_, MAT_INITIAL_MATRIX, PETSC_DEFAULT,&BTGpetsc_);
    //C=A*B.
    MatMatMult(BTGpetsc_,Bpetsc_, MAT_INITIAL_MATRIX , PETSC_DEFAULT, &AnearOptiPETSc_);
    // y = Ax
    MatMult(BTGpetsc_,Fpetsc_,lnearOptiPETSc_);


    //cout<<"this is AnearOptiPETSc_"<<endl;
    //MatView(AnearOptiPETSc_,0);

    MatDestroy(&Bpetsc_);
    MatDestroy(&Gpetsc_);
    MatDestroy(&BTGpetsc_);
    VecDestroy(&Fpetsc_);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
             Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    KSPCreate(PETSC_COMM_WORLD,&ksp_);

/*
   Set operators. Here the matrix that defines the linear system
   also serves as the matrix that defines the preconditioner.
*/
    KSPSetOperators(ksp_,AnearOptiPETSc_,AnearOptiPETSc_);

/*
   Set linear solver defaults for this problem (optional).
   - By extracting the KSP and PC contexts from the KSP context,
     we can then directly call any KSP and PC routines to set
     various options.
   - The following four statements are optional; all of these
     parameters could alternatively be specified at runtime via
     KSPSetFromOptions();
*/
    KSPGetPC(ksp_,&pc_);
    //PCSetType(pc_,PCJACOBI);
    KSPSetTolerances(ksp_,1.e-10,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
////---- tolerance 10^-16 is very perfect but it is very slow!!!
/*
  Set runtime options, e.g.,
      -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
  These options will override those specified above as long as
  KSPSetFromOptions() is called _after_ any other customization
  routines.
*/
    KSPSetFromOptions(ksp_);

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Solve the linear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    KSPSolve(ksp_,lnearOptiPETSc_,Sol_);

/*
   View solver info; we could instead use the option -ksp_view to
   print this info to the screen at the conclusion of KSPSolve().
*/
   // KSPView(ksp_,PETSC_VIEWER_STDOUT_WORLD);


    MatDestroy(&AnearOptiPETSc_);
    VecDestroy(&lnearOptiPETSc_);

    // Uinital_ = Sol_;
    PETScVectoEigenVec(Sol_, m_, Uinital_);

    //VecView(Sol_,0);
    VecDestroy(&Sol_);
    KSPDestroy(&ksp_);

}
void primalEuropeanOptions::BackwardEulerSolver()
{

    setInitialValue();

    for(int i=1;i<=Nt_;i++)
    {
        cout<<"Nelems " <<Nelems_<<", Time Step of Primal DPG-->:"<<i<<"/"<<Nt_<<endl;

        assembly();

        optionValue();
        if (optionType_==1)//0:European, 1:American
        {
          imposeFreeBoundary();
        }


        decomposeSol();
    }

    decomposeSol();
    //cout<<"This is trial"<<endl;
    //cout<<Utrial_;

    PetscFinalize();
}
void primalEuropeanOptions::savePrimalSolution()
{
    ofstream file1("trialSoltionPriamlDPG.txt");

    if(!file1.is_open() )
    {
        cout<<"unable to open files"<<endl;
    }else

    {
        //Saving numerical solution
        for(int i=0; i<Utrial_.size();i++)
        {
            file1 << Utrial_(i)  << endl;

        }

        file1.close();


    }


}
void primalEuropeanOptions::saveResults()

{
    ofstream file1("trialSoltionPriamlDPG.dat");
    ofstream file2("GlobalRawErrorPriamlDPGR05sigma03Nelem200P1.dat");
    if(!file1.is_open() )
    {
        cout<<"unable to open files"<<endl;
    }else

        {
        //Saving numerical solution
        for(int i=0; i<Utrial_.size();i++)
        {
            file1 << Utrial_(i)  << endl;
            file2 << generalError(i)  << endl;
        }

        file1.close();
        file2.close();

        }
    ofstream file3("GlobalL2ErrorPriamlDPG.dat");
    ofstream file4("GlobalLinfErrorPriamlDPG.dat");
    file3 << errorL2<< endl;
    file4 << errorLinf<< endl;
    file3.close();
    file4.close();

}

void primalEuropeanOptions::findError()
{

    VectorXd solExact_(Nnodes_);
    VectorXd err1_;

    double normL2error_;
    double normLinferror_;
    double aBSerrorL2_;
    double aBSerrorLinf_;
    BackwardEulerSolver();
    if (optionType_==0)
    {
        for(int i=0;i<=Nnodes_-1;i++)
        {
            cout<<"Binomial method for European value at node -->: "<<i<<"/"<<Nnodes_<<endl;
            solExact_(i) = Euro9PutDesHigham(exp(x_(i)));
        }
    } else
      {
        cout << "Binomial method for American option" << endl;
        solExact_ = AmericanPutDesHigham(Nnodes_);
    }

     //cout<<solExact_<<endl;
    err1_         = Utrial_ - solExact_;
    aBSerrorL2_   = (err1_.array().square()).sum();
    aBSerrorLinf_ = (err1_.array().abs()).maxCoeff();
    normL2error_  = sqrt(aBSerrorL2_/((solExact_.array().square()).sum()));
    normLinferror_= aBSerrorLinf_/((solExact_.array().abs()).maxCoeff());


    errorL2       = normL2error_;
    errorLinf     = normLinferror_;
    generalError  = err1_;
    cout<<"errorL2    "<<errorL2<<endl;
    cout<<"errorLinf  "<<errorLinf<<endl;

    saveResults();

}

double primalEuropeanOptions::Euro9PutDesHigham(float x)
{
    ///------ BS Black-Scholes European put price
    float S_ = x;
    float d1_ = (log(S_/k_) + (r_ +0.5 * sigma_*sigma_)*T_)/sigma_*sqrt(T_);
    float d2_ = d1_-sigma_ *sqrt(T_);
    double N1_ = 0.5 *(1+erf(-d1_/sqrt(2.0)));
    double N2_ = 0.5 *(1+erf(-d2_/sqrt(2.0)));


    return k_*exp(-r_*T_)*N2_-S_*N1_;
}


void primalEuropeanOptions::imposeFreeBoundary() {

  VectorXd Xsol;
  Xsol = VectorXd::Zero(Ntdof_);
  float ax1_;
  float ax2_;
  float ax3_;

  switch (contractType_) {
  case 0: // 0: call,
  {

    for (int i = 0; i <= Nnodes_ - 1; i++) {

      ax1_ = exp(x_(i)) - k_;

      Xsol(i) = max(ax1_, 0.0f);

      ax2_ = Xsol(i);

      ax3_ = Uinital_(i);

      Uinital_(i) = max(ax2_, ax3_);
    }

    break;
  }

  case 1: // 1:put
  {
    for (int i = 0; i <= Nnodes_ - 1; i++)
    {

      ax1_ = k_ - exp(x_(i));
      Xsol(i) = max(ax1_, 0.0f);
      ax2_ = Xsol(i);
      ax3_ = Uinital_(i);
      Uinital_(i) = max(ax2_, ax3_);
    }

    break;
  }
  }
}

///--------------------------American-----------------

VectorXd primalEuropeanOptions::AmericanPutDesHigham(int x)
    {
///------ AMERICAN Binomial method for an American put
///------ Vectorized version, based on euro5.m


    string tmp;
    int cnt = 0;
    double tempInt;
    VectorXd Y_ = VectorXd::Zero(x);



///-------- Reading Exact solutions inot Eigen objects--------
    ifstream file5("exactSolution/exactsoluitonR05sigma03Nelem200P1.dat");

    //ifstream file5(tmp);
   // vector<float> vec;
    if(!file5.is_open() )
    {
    cout<<"unable to open files"<<endl;
    } else
      {
      while (file5>>tempInt)
      {
        Y_(cnt) = tempInt;
        //cout<<"I am here"<<endl<<Y_(cnt)<<endl;
        cnt += 1;
      }
    }

       cout<<Y_<<endl;
       return Y_;
    }