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
#include "Eigen/Sparse"
#include <petscksp.h>
#include "petscmat.h"

#include "ultraweakEuropeanOptions.h"

static char help[] = "Solves a tridiagonal linear system with KSP.\n\n";

using namespace std;
using namespace Eigen;
#define PI 3.14159265
///-------------- Implementation of Constructor---------
ultraweakEuropeanOptions::ultraweakEuropeanOptions()
{
    Nnodes_ = trilDim_ * Nelems_ + 1 ;
    length_ = A1_- A0_   ;
    h_      = length_ /Nelems_;
    x_      = VectorXd::LinSpaced(Nnodes_,A0_,A1_);
    dxdz_   = h_/2.0;
    cof1_   = 0.5 * pow(sigma_,2);
    cof2_   = r_ - 0.5 * pow(sigma_,2);
    cof3_   = r_;

    //Ntdof_  = 2.0 * Nnodes_;        //I guess this is total Dof //This is right just for PRIMAL!!!!

    NU_     = trilDim_ + 1 ; //trial Dof
    NV_     = testDim_ + 1;  //test Dof
    Nuh_    = tracDim_;      //trace Dof???????????????

    locDof_ = NU_*Nelems_;
    base_   = 2*locDof_ + 1;
    //shift_  = 2*Nuh_ * Nelems_+2;
    n_      = 2*NV_*Nelems_;
    m_      = 2*locDof_ + 2*Nuh_ * Nelems_+2;



    PetscInitialize(&argc,&args,(char*)0,help);
    comm_ = PETSC_COMM_WORLD;
    MPI_Comm_rank(comm_,&rank_);
}

///-------------- Implementation of Public Method---------
void ultraweakEuropeanOptions::run()
{

    //findInitialValue();
    //setInitialValue();
    //decomposeSol();
    //cout<<Uinital_<<endl;
    //decomposeSol();
    //lgwt(3,-1,1);
    //assembly();
    //optionValue();
    BackwardEulerSolver();
    //saveTotalSolution();
    //postProcessingTrial();

}

void ultraweakEuropeanOptions::BackwardEulerSolver()
{
    findInitialValue();
    setInitialValue();

    for (int i = 1; i <= Nt_; i++)
    {
        cout << "Nelems " << Nelems_ << ", Time Step of ultraweak DPG-->:" << i << "/" << Nt_ << endl;
        assembly();

        optionValue();

        if (optionType_==1)//0:European, 1:American
        {
            imposeFreeBoundary();
        }
        decomposeSol();
    }
    decomposeSol();
    postProcessingTrial();
    cout<<Utrial_<<endl;
    PetscFinalize();
}
void ultraweakEuropeanOptions::findError()
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
        cout<<"Binomial method for vlaue at node -->: "<<i<<"/"<<Nnodes_<<endl;
        solExact_(i) = Euro9PutDesHigham(exp(x_(i)));

    }
    } else
    {
        cout << "Binomial method for American option" << endl;
        solExact_ = AmericanPutDesHigham(Nnodes_);
    }

    err1_         = Utrial_ - solExact_;
    aBSerrorL2_   = (err1_.array().square()).sum();
    aBSerrorLinf_ = (err1_.array().abs()).maxCoeff();
    normL2error_  = sqrt(aBSerrorL2_/((solExact_.array().square()).sum()));
    normLinferror_= aBSerrorLinf_/((solExact_.array().abs()).maxCoeff());


    errorL2       = normL2error_;
    errorLinf     = normLinferror_;
    generalError  = err1_;
    cout<<"errorL2   "<<errorL2<<endl;
    cout<<"errorLinf  "<<errorLinf<<endl;

    saveResults();
}

///-------------- Implementation of Private Method---------
MatrixXd  ultraweakEuropeanOptions::myTranspose(VectorXd vec)
{
    int sizeVec_  = vec.size();

    MatrixXd vecT_(1,sizeVec_ );

    for(int i=0; i<sizeVec_;i++)
    {
        vecT_(0,i) = vec(i);
    }

    return vecT_;
}
void ultraweakEuropeanOptions::PETScVectoEigenVec(Vec& pVec, int n, VectorXd& eVec)
{
      double vali_ ;
      for(int row=0; row<n;row++)
      {
            VecGetValues(pVec,1,&row, &vali_);
             eVec(row) = vali_;
      }

}
void ultraweakEuropeanOptions::lgwt(int dim_, double a, double b)
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

void ultraweakEuropeanOptions::findInitialValue()
{
    VectorXd Xsol_;
    Xsol_ = VectorXd::Zero(locDof_);

    float ax1_;
    int xIdx;
    int a1_;int b1_;int a2_;int b2_;
    for(int i=1;i<=Nelems_;i++)
    {
        a1_ = (NU_-1) * i - ((NU_-1)-1)-1;
        b1_ = (NU_-1) * i + 1 - 1 ;

        a2_ = NU_ * (i-1) + 1 - 1;
        b2_ = NU_ * i-1;
        switch (contractType_)
        {
            case 0:          //0: call,
            {
                xIdx = a1_;
                for(int j=a2_;j<=b2_;j++)
                {

                    ax1_ = exp(x_(xIdx)) - k_;

                    Xsol_(j) = max(ax1_, 0.0f);
                    xIdx+=1;
                }
                Ulocinital_ = Xsol_;
                break;
            }

            case 1:         // 1:put
            {
                xIdx = a1_;
                for(int j=a2_;j<=b2_;j++)
                {

                   // cout<<" is "<<j<<endl;
                    ax1_ = k_ - exp(x_(xIdx));
                    Xsol_(j) = max(ax1_, 0.0f);
                    xIdx += 1;
                }
                Ulocinital_ = Xsol_;
                break;
            }

        }

    }

}


void ultraweakEuropeanOptions::setInitialValue()
{

    VectorXd Xsol_;
    Xsol_ = VectorXd::Zero(m_);
    float ax2_;
    int xIdx;
    int a1_;int b1_;int a2_;int b2_;
    int locBase_; int totalIdx;
    for(int i=1;i<=Nelems_;i++)
    {
        a1_ = NU_ * (i - 1)+ 1 - 1;
        b1_ = NU_ * i - 1 ;
        locBase_ = NU_*(i-1);
        a2_ = locBase_ + NU_ * (i-1) + 1 - 1;
        b2_ = locBase_ + NU_ * i - 1;
        xIdx = a1_;
        for(int j=a2_;j<=b2_;j++)
        {

            Xsol_(j) = Ulocinital_(xIdx);
            xIdx += 1;
        }

        }
    Uinital_ = Xsol_;
}

void ultraweakEuropeanOptions::imposeFreeBoundary()
{
    VectorXd Xsol_;
    Xsol_ = VectorXd::Zero(m_);

    int xIdx;
    int a1_;int a2_;int b2_;
    int locBase_;
    float ax2_;
    float ax3_;

    for(int i=1;i<=Nelems_;i++)
    {
        a1_ = NU_ * (i - 1)+ 1 - 1;
        locBase_ = NU_*(i-1);
        a2_ = locBase_ + NU_ * (i-1) + 1 - 1;
        b2_ = locBase_ + NU_ * i - 1;
        xIdx = a1_;
        for(int j=a2_;j<=b2_;j++)
        {

            Xsol_(j) = Ulocinital_(xIdx);
            ax2_ = Xsol_(j);
            ax3_ = Uinital_(j);
            Uinital_(j) = max(ax2_,ax3_);
            xIdx += 1;
        }

    }

}
void ultraweakEuropeanOptions::decomposeSol()
{
    /// needs to be pullised!!!!!!!!!!!!
    Utotal_ = Uinital_;


    uVec_   = VectorXd::Zero(locDof_ );
    sigVec_ = VectorXd::Zero(locDof_ );
    uHat_   = VectorXd::Zero(Nuh_ * Nelems_+1);
    qnHat_  = VectorXd::Zero(Nuh_ * Nelems_+1);

    int a1_;int a2_;int a3_;int a4_;
    int baseInx1_;int baseInx2_ = 2*locDof_;
    int baseInx3_ = 2*locDof_ + Nuh_ * Nelems_+1;
    int idx_1;int idx_2;int idx_3;int idx_4;
    int trialIdx_;int testIdx_;int trialIdx1_;

    baseInx2_ = 2 * NU_ * Nelems_;
    baseInx3_ = 2 * NU_ * Nelems_ + Nuh_*Nelems_ + 1;

    for(int i=1;i<=Nelems_;i++)
    {

        trialIdx_ = NU_ * (i - 1);
        testIdx_  = (i-1);

        a1_ = 2 * NU_ * (i - 1) + 1 - 1;
        a2_ =  a1_ + NU_;

        a3_ = baseInx2_ + (i-1) ;
        a4_ = baseInx3_ + (i-1) ;

        for(int j=1;j<=NU_;j++)
        {
            idx_1 = a1_ + (j-1);
            idx_2 = a2_ + (j-1);
            trialIdx1_ = trialIdx_+(j-1);
            uVec_(trialIdx1_)  = Utotal_(idx_1);
            sigVec_(trialIdx1_)= Utotal_(idx_2);
        }
            uHat_(testIdx_)  = Utotal_(a3_);
            qnHat_(testIdx_) = Utotal_(a4_);
    }

    uHat_(testIdx_+1)  = Utotal_(a3_+1);    //last data of the last element
    qnHat_(testIdx_+1) = Utotal_(a4_+1);
/*
    cout<<"uVec_"  <<endl<<uVec_<<endl;
    cout<<"sigVec_"<<endl<<sigVec_<<endl;
    cout<<"uHat_"  <<endl<<uHat_<<endl;
    cout<<"qnHat_" <<endl<<qnHat_<<endl;
    */
}

void ultraweakEuropeanOptions::assembly()
{

    ///-------- PETSc init-----------
    //preparing some stuff for petsc
    int indU_[2*NU_];
    int indV_[2*NV_];
    int indUhat_[Nuh_+1];
    int indQhat_[Nuh_+1];
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

    MatrixXd tempGe_      = MatrixXd::Zero(2*NV_,2*NV_);
    MatrixXd tempBe_      = MatrixXd::Zero(2*NV_,2*NU_);
    MatrixXd tempBhate_   = MatrixXd::Zero(2*NV_,Nuh_+1);
    MatrixXd tempBqhate_  = MatrixXd::Zero(2*NV_,Nuh_+1);
    MatrixXd tempGeT_     = MatrixXd::Zero(2*NV_,2*NV_);
    MatrixXd tempBeT_     = MatrixXd::Zero(2*NU_,2*NV_);
    MatrixXd tempBhateT_  = MatrixXd::Zero(Nuh_+1,2*NV_);
    MatrixXd tempBqhateT_ = MatrixXd::Zero(Nuh_+1,2*NV_);
    VectorXd tempFe_      = VectorXd::Zero(2*NV_);

    ///---- Initialize the PETSc obj -------

    MatZeroEntries(Gpetsc_);
    MatZeroEntries(Bpetsc_);
    VecZeroEntries(Fpetsc_);

   ///-------- PETSc init-----------END


    MatrixXd axB_(n_,m_);
    MatrixXd axG_(n_,m_);
    VectorXd axF_(n_);


    MatrixXd A1mat_;
    MatrixXd A2mat_;
    MatrixXd A3mat_;
    MatrixXd A4mat_;
    MatrixXd A5mat_;
    MatrixXd A6mat_;

    MatrixXd d1mat_(2*NV_,Nuh_+1);
    MatrixXd d2mat_(2*NV_,Nuh_+1);
    MatrixXd Ge_(2*NV_,2*NV_);

    VectorXd Fe1_(2*NV_);
    VectorXd Fe2_(2*NV_);
    VectorXd Fe3_(2*NV_);
    VectorXd Fe4_(2*NV_);
    VectorXd Fe5_(2*NV_);
    VectorXd idxU_;
    VectorXd idxV_;
    VectorXd idxUh_;
    VectorXd idxQh_;
    VectorXd N_;
    VectorXd dN_dz_;
    VectorXd M_;
    VectorXd dM_dz_;

    int a1; int a2;int b1;int b2;int a3;int b3;int a4;int b4;

    int baseIdx;int baseUhat;int shift_;
    int idx1_;int idx2_;


    ///---- initializing-----

    axB_ = MatrixXd::Zero(n_,m_);
    axG_ = MatrixXd::Zero(n_,n_);
    axF_ = VectorXd::Zero(n_);

    ///----- quad values-----

    lgwt(quadPoints,-1,1);

    decomposeSol();

    for(int i=1; i<=Nelems_;i++)
    {
        A1mat_ = MatrixXd::Zero(2*NV_,2*NU_);
        A2mat_ = MatrixXd::Zero(2*NV_,2*NU_);
        A3mat_ = MatrixXd::Zero(2*NV_,2*NU_);
        A4mat_ = MatrixXd::Zero(2*NV_,2*NU_);
        A5mat_ = MatrixXd::Zero(2*NV_,2*NU_);
        A6mat_ = MatrixXd::Zero(2*NV_,2*NU_);
        d1mat_ = MatrixXd::Zero(2*NV_,Nuh_+1); d1mat_(NV_,0) = -1.0;d1mat_(2*NV_-1,Nuh_) = 1.0;
        d2mat_ = MatrixXd::Zero(2*NV_,Nuh_+1); d2mat_(0,0) = -1.0;d2mat_(NV_-1,Nuh_) = 1.0;

        Ge_    = MatrixXd::Zero(2*NV_,2*NV_);

        Fe1_   = VectorXd::Zero(2*NV_);

        ///-------Indexing for test and trial variables ------------
        a1 = 2*NU_*(i-1)+1-1;
        b1 = 2*NU_*i-1;
        a2 = 2*NV_*(i-1)+1-1;
        b2 = 2*NV_*i - 1 ;
        idxU_  = VectorXd::LinSpaced(b1-a1+1,a1,b1);


        idxV_  = VectorXd::LinSpaced(b2-a2+1,a2,b2);
        ///-------Indexing for trace and flux variables ------------
        baseIdx = 2* Nelems_ * NU_-1;
        shift_ = Nuh_ * Nelems_ + 1;


        a3 = baseIdx + Nuh_ * i - (Nuh_-1)-1;
        b3 = baseIdx + Nuh_ * i + 1 -1;
        a4 = baseIdx + shift_+ Nuh_ * i - (Nuh_-1)-1;
        b4 = baseIdx + shift_+ Nuh_ * i + 1 -1;

        idxUh_ = VectorXd::LinSpaced(b3-a3+1,a3,b3);
        idxQh_ = VectorXd::LinSpaced(b4-a4+1,a4,b4);

        ///-------Indexing for Uold-------
        idx1_ = NU_*(i-1); idx2_ = NU_*i-1;

        for(int j = 0;j<=quadPoints-1;j++)
        {
            N_     = feTrail.shapVal(qpLoc(j));
            M_     = feTest.shapVal(qpLoc(j));
            dM_dz_ = feTest.shapDVal(qpLoc(j))  * 1.0/dxdz_;
            ///--------Forming Bilinear form (Ultraweak)---------
            //P.block(i, j, rows, cols)          // P(i+1 : i+rows, j+1 : j+cols)
            //P.block(0, 0, NV, NU)          // P(1 : NV, 1 : NU)
            //P.block(0, NU, NV, 2*NU-NU)          // P(1:NV,NU+1:2*NU)
            //P.block(NV,0 ,2*NV-NV, NU)          // P(NV+1:2*NV,1:NU)


            //(u_{n+1},v)_Ii
            A1mat_.block(0, 0, NV_, NU_) += qpWgt(j) * M_ * N_.transpose() * dxdz_ ;
            // (sig , c1v')_Ii
            A2mat_.block(0, NU_, NV_, 2*NU_-NU_) += qpWgt(j) * cof1_ * dM_dz_ * N_.transpose() * dxdz_ ;
            // (sig, c2v)_Ii
            A3mat_.block(0, NU_, NV_, 2*NU_-NU_) += qpWgt(j) * cof2_ * M_ * N_.transpose() * dxdz_ ;
            //(u,rv)_Ii
            A4mat_.block(0, 0, NV_, NU_) += qpWgt(j) *cof3_* M_ * N_.transpose() * dxdz_ ;
            //(u,t')_Ii
            A5mat_.block(NV_,0 ,2*NV_-NV_, NU_) += qpWgt(j) * dM_dz_ * N_.transpose() * dxdz_ ;
            // (sig, t)_Ii
            A6mat_.block(NV_,NU_ ,2*NV_-NV_, 2*NU_-NU_)+=qpWgt(j) * M_ * N_.transpose() * dxdz_ ;


            ///----------- Forming Gram Matrix---------------
            //(c2v',c2v')
            Ge_.block(0,0 ,NV_, NV_)+=qpWgt(j) * cof2_* cof2_ * dM_dz_ * dM_dz_.transpose() * dxdz_ ;
            //(c2v',-c3v)
            Ge_.block(0,0 ,NV_, NV_)+=qpWgt(j) * -cof2_* cof3_ * dM_dz_ * M_.transpose() * dxdz_ ;
            //(c2v',-t)
            Ge_.block(0,NV_ ,NV_, 2*NV_-NV_)+=qpWgt(j) * -cof2_* cof3_ * dM_dz_ * M_.transpose() * dxdz_ ;
            //(-c3v,c2v')
            Ge_.block(0,0 ,NV_, NV_)+=qpWgt(j) * -cof2_* cof3_ * M_ * dM_dz_.transpose() * dxdz_ ;
            //(-c3v,-c3v)
            Ge_.block(0,0 ,NV_, NV_)+=qpWgt(j) * -cof3_* cof3_ * M_ * M_.transpose() * dxdz_ ;
            //(-c3v,-t)
            Ge_.block(0,NV_ ,NV_, 2*NV_-NV_)+=qpWgt(j) * -cof3_* M_ * -M_.transpose() * dxdz_ ;
            //(-t,c2v')
            Ge_.block(NV_,0 ,2*NV_-NV_, NV_)+=qpWgt(j) * -cof2_* M_ * -dM_dz_.transpose() * dxdz_ ;
            //(-t,-c3v)
            Ge_.block(NV_,0 ,2*NV_-NV_, NV_)+=qpWgt(j) * -cof3_* M_ * -M_.transpose() * dxdz_ ;
            //(-t,-t)          1+a??????????????
            Ge_.block(NV_,NV_ ,2*NV_-NV_, 2*NV_-NV_)+=qpWgt(j) * -M_ * -M_.transpose() * dxdz_ ;
            ///------
            //(c1v-t',c1v-t')
            //
            //(c1v,c1v)   1+a
            Ge_.block(0,0 ,NV_, NV_)+=qpWgt(j) * cof1_ * M_ * cof1_ * M_.transpose() * dxdz_ ;
            //(c1v,-t')
            Ge_.block(0,NV_  ,NV_, 2*NV_-NV_)+=qpWgt(j) * cof1_ * M_ * -dM_dz_.transpose() * dxdz_ ;
            //(-t',c1v)
            Ge_.block(NV_,0 ,2*NV_-NV_,NV_ )+=qpWgt(j) * cof1_ * -dM_dz_ * M_.transpose() * dxdz_ ;
            //(-t',-t')                1+a
            Ge_.block(NV_,NV_ ,2*NV_-NV_,2*NV_-NV_ )+=qpWgt(j) * -dM_dz_ * -dM_dz_.transpose() * dxdz_ ;
            ///----------Forming RHS -------------------------
            //x.segment(i, n)                    // x(i+1 : i+n)
            Fe1_.segment(0,NV_)+=  qpWgt(j) *  M_ * (uVec_.segment(idx1_,idx2_-idx1_+1).transpose() * N_)  * dxdz_;


            ///------------Theta Coefficients---------------



        }

        ///--------- PETSc Objects ------------------
        // Preparing Indexes for PETSc
        for(int iu = 0;iu<=2*NU_-1;iu++)
        {
            indU_[iu] = a1 + iu ;
        }
        for(int iv = 0;iv<=2*NV_-1;iv++)
        {
            indV_[iv] = a2 + iv ;
        }
        for(int iuh = 0;iuh<=Nuh_;iuh++)///????????????
        {
            indUhat_[iuh] = a3 + iuh + 1 ;
        }
        for(int iqh = 0;iqh<=Nuh_;iqh++)///????????????
        {
            indQhat_[iqh] = a4 + iqh + 1 ;
        }
        tempGe_       = Ge_.inverse();
        tempBe_       = A1mat_-teta_*dt_*(-A2mat_ + A3mat_ - A4mat_) - A5mat_ - A6mat_;
        tempBhate_    = d1mat_;
        tempBqhate_   =  - dt_* d2mat_;
        tempFe_       = Fe1_;//+(1.0-teta_)*dt_*(-Fe2_+Fe3_-Fe4_);  //teta method!!!!

        tempGeT_      = tempGe_.transpose();
        tempBeT_      = tempBe_.transpose();
        tempBhateT_   = tempBhate_.transpose();
        tempBqhateT_  = tempBqhate_.transpose();
        double *dataF = tempFe_.data();

        ////-------??????????for ultraweak formualtion pay attention to INSERT_VALUES or add_values!!

        MatSetValues(Gpetsc_,2*NV_,indV_,2*NV_,indV_,tempGeT_.data(),INSERT_VALUES);
        MatSetValues(Bpetsc_,2*NV_,indV_,2*NU_,indU_,tempBeT_.data(),INSERT_VALUES);
        MatSetValues(Bpetsc_,2*NV_,indV_,Nuh_+1,indUhat_,tempBhateT_.data(),INSERT_VALUES);
        MatSetValues(Bpetsc_,2*NV_,indV_,Nuh_+1,indQhat_,tempBqhateT_.data(),INSERT_VALUES);
        //VecSetValues(Fpetsc_,NV_,indV_,tempFe_.data(),INSERT_VALUES);
        for(int i=0;i<2*NV_;i++)
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



}
void ultraweakEuropeanOptions::optionValue()
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


void ultraweakEuropeanOptions::postProcessingTrial()
{
    int endY = Nelems_*(NU_-1)+1;
    VectorXd Y_ = VectorXd::Zero(endY);
    Y_(0) = uVec_(0);
    Y_(endY-1) = uVec_(Nelems_*NU_-1);
    int cnt  = 1;
    int idxZ = 0;

    for(int i=1;i<endY-1;i++)
    {
        idxZ +=1;
        if (cnt==NU_-1)
        {
            Y_(i) = (uVec_(idxZ)+uVec_(idxZ+1))/2;
            idxZ +=1;
            cnt = 1;
        }
        else
        {
            Y_(i)= uVec_(idxZ);
            cnt += cnt;
        }
    }
    Utrial_=Y_;



}
void ultraweakEuropeanOptions::saveTotalSolution()
{
    ofstream file1("trialSoltionPriamlDPG.txt");

    if(!file1.is_open() )
    {
        cout<<"unable to open files"<<endl;
    }else

    {
        //Saving numerical solution
        for(int i=0; i<Utotal_.size();i++)
        {
            file1 << Utotal_(i)  << endl;

        }

        file1.close();


    }


}


void ultraweakEuropeanOptions::saveResults()
{
    ofstream file1("trialSoltionUltraweakDPG.dat");
    ofstream file2("GlobalRawErrorUltraweakDPGN10r05sig015p2.dat");
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
    file3 << errorL2  << endl;
    file4 << errorLinf  << endl;
    file3.close();
    file4.close();

}
double ultraweakEuropeanOptions::Euro9PutDesHigham(float x)
{
    ///------ BS Black-Scholes European put price
    float S_ = x;
    float d1_ = (log(S_/k_) + (r_ +0.5 * sigma_*sigma_)*T_)/sigma_*sqrt(T_);
    float d2_ = d1_-sigma_ *sqrt(T_);
    double N1_ = 0.5 *(1+erf(-d1_/sqrt(2.0)));
    double N2_ = 0.5 *(1+erf(-d2_/sqrt(2.0)));


    return k_*exp(-r_*T_)*N2_-S_*N1_;
}

VectorXd ultraweakEuropeanOptions::AmericanPutDesHigham(int x)
{
///------ AMERICAN Binomial method for an American put
///------ Vectorized version, based on euro5.m

    string tmp;
    int cnt = 0;
    double tempInt;
    VectorXd Y_ = VectorXd::Zero(x);

///-------- Reading Exact solutions inot Eigen objects--------
    ifstream file5("exactSolution/p2/exactsoluitonR05sigma012Nelem200P2.dat");

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