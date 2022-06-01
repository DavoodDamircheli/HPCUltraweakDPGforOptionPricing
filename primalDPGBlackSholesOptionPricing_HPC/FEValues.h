//
// Created by Davood Damirchelli on 2021-10-18.
//
///------------Goal--------------
/// a class for providing value of FEM space
///-------------------------------------

#include <iostream>
#include <fstream>
#include "Eigen/Dense"
#include "Eigen/LU"
#include <math.h>
#include <vector>


#ifndef PRIMALDPG_FEVALUES_H
#define PRIMALDPG_FEVALUES_H

using namespace std;
using namespace Eigen;

template<int T>
class FEValues
{

public:

    VectorXd  zc;
    MatrixXd  N;
    MatrixXd  dNdz;
    MatrixXd  d2Ndz2;

    ///constructor
    //FEValues();
    FEValues();
    ///-------Public Methods------------
    MatrixXd & shapVal(double qp);
    MatrixXd & shapDVal(double qp);
private:

    int a_;
    double qp_;

    MatrixXd  CoeffN;
    MatrixXd  CoeffdNdz;

    ///----- Private Methods----------
    void shapCoeffV(int dim);
    void shapCoeffVD(int dim);
};


template<int T>
FEValues<T>::FEValues()
{
 a_  = T;

}

template<int T>
void FEValues<T>::shapCoeffVD(int dim)
{
    switch (dim)
    {

        case 1:
        {

            VectorXd  dNdz1(2);
            dNdz1   << -0.5, 0.5;

            CoeffdNdz = dNdz1;

            break;
        }
        case 2:
        {
            MatrixXd  dNdz1(3,2);


            dNdz1   << 1.0, -0.5,
                    -2.0,0.0,
                    1.0,0.5;

            CoeffdNdz = dNdz1;

            break;
        }
        case 3:
        {

            MatrixXd  dNdz1(4,3);
                       dNdz1   <<  -1.6875,    1.1250,    0.0625,
                                   5.0625,   -1.1250,   -1.6875,
                                  -5.0625,   -1.1250,   1.6875,
                                   1.6875,    1.1250,   -0.0625;
            CoeffdNdz=dNdz1;
            break;
        }
        case 4:
        {

            MatrixXd  dNdz1(5,4);



            dNdz1   <<   2.6667,   -2.0000,   -0.3333,    0.1667,
                    -10.6667,    4.0000,    5.3333,   -1.3333,
                    16.0000,         0,  -10.0000,         0,
                    -10.6667,   -4.0000,    5.3333,    1.3333,
                    2.6667,    2.0000,   -0.3333,   -0.1667;
            CoeffdNdz=dNdz1;
            break;
        }
        case 5:
        {

            MatrixXd  dNdz1(6,5);


            dNdz1   <<-4.0690 ,   3.2552,    0.9766,   -0.6510,   -0.0117,
                    20.3451,   -9.7656 , -12.6953,    5.0781 ,   0.1628,
                    -40.6901,    6.5104 ,  33.2031,   -4.4271,   -2.9297,
                    40.6901 ,   6.5104,  -33.2031,   -4.4271,    2.9297,
                    -20.3451,   -9.7656 ,  12.6953,    5.0781,   -0.1628,
                    4.0690,    3.2552,   -0.9766,   -0.6510,    0.0117;
            CoeffdNdz=dNdz1;
        break;
        }
        case 6:
        {
            MatrixXd  dNdz1(7,6);

            dNdz1   <<    6.0750,   -5.0625,   -2.2500,    1.6875,    0.1000,   -0.0500,
                    -36.4500,   20.2500,   27.0000,  -13.5000,   -1.3500,    0.4500,
                    91.1250,  -25.3125,  -87.7500,   21.9375,   13.5000,   -2.2500,
                    -121.5000,         0.00,  126.0000,         0.00,  -24.5000,0.00,
                    91.1250,   25.3125,  -87.7500,  -21.9375,   13.5000,    2.2500,
                    -36.4500,  -20.2500,   27.0000 ,  13.5000,   -1.3500,   -0.4500,
                    6.0750,    5.0625,   -2.2500,   -1.6875,    0.1000,    0.0500;
            CoeffdNdz=dNdz1;
            break;
        }


    }
}

///-------- Implementing Private Method----------
template<int T>
void FEValues<T>::shapCoeffV(int dim)
{

    switch (dim)
    {

        case 1:
            {
                MatrixXd  N1(2,2);

                N1      <<-0.5,0.5,
                           0.5,0.5;

                CoeffN = N1;
            break;
            }
        case 2:
        {


            MatrixXd  N1(3,3);

            N1     <<0.5,-0.5,0.0,
                    -1.0,0.0,1.0,
                    0.5,0.5,0.0;

            CoeffN = N1;

            break;
        }
        case 3:
        {

            MatrixXd  N1(4,4);

            N1      << -0.5625,    0.5625,    0.0625,   -0.0625,
                         1.6875,   -0.5625,   -1.6875,    0.5625,
                         -1.6875,   -0.5625,    1.6875,    0.5625,
                        0.5625,    0.5625,   -0.0625,   -0.0625;

            CoeffN = N1;
            break;
        }
        case 4:
        {

            MatrixXd  N1(5,5);

            N1      <<  0.6667,   -0.6667,   -0.1667,    0.1667 ,        0,
                    -2.6667 ,   1.3333,    2.6667,   -1.3333 ,        0,
                    4.0000 ,        0,   -5.0000,         0,    1.0000,
                    -2.6667,   -1.3333,    2.6667,    1.3333,         0,
                    0.6667,    0.6667,   -0.1667,   -0.1667,         0;
            CoeffN = N1;
            break;
        }
        case 5:
        {

            MatrixXd  N1(6,6);



            N1      <<-0.8138,    0.8138 ,   0.3255 ,  -0.3255,   -0.0117 ,   0.0117,
                    4.0690,   -2.4414,   -4.2318,    2.5391 ,   0.1628,   -0.0977,
                    -8.1380,    1.6276,   11.0677,   -2.2135,   -2.9297,    0.5859,
                    8.1380,    1.6276,  -11.0677,   -2.2135,    2.9297,    0.5859,
                    -4.0690,   -2.4414,    4.2318,    2.5391,   -0.1628,   -0.0977,
                    0.8138,    0.8138,   -0.3255,   -0.3255,    0.0117,    0.0117;

            CoeffN = N1;
            break;
        }

        case 6:
        {
            MatrixXd  N1(7,7);

            N1      <<  1.0125,   -1.0125,   -0.5625 ,   0.5625 ,   0.0500,   -0.0500 ,        0.0000,
                    -6.0750 ,   4.0500,    6.7500,   -4.5000,   -0.6750,    0.4500,         0.0000,
                    15.1875,   -5.0625,  -21.9375,    7.3125,    6.7500,   -2.2500,         0.0000,
                    -20.2500,         0.0000,   31.5000,         0.0000,  -12.2500 ,        0.0000,    1.0000,
                    15.1875,    5.0625,  -21.9375 ,  -7.3125,    6.7500 ,   2.2500,         0.0000,
                    -6.0750,   -4.0500,    6.7500,    4.5000,   -0.6750 ,  -0.4500,         0.0000,
                    1.0125,    1.0125,   -0.5625,   -0.5625 ,   0.0500 ,   0.0500 ,        0.0000;


            CoeffN = N1;
            break;
        }


    }
}


template<int T>
MatrixXd& FEValues<T>::shapVal(double qp)
{

    double   dxdz;
    //a_ = b;
    qp_ = qp;

    shapCoeffV(a_);  // cofN ,
    MatrixXd N1;
    N1      = MatrixXd::Zero(a_+1,1);

    for(int i=0;i<=a_;i++)
    {
         N1    =  N1 + CoeffN.col(i)* pow(qp_,a_-(i));
    }
    N = N1;
    return N;
}

template<int T>
MatrixXd& FEValues<T>::shapDVal(double qp)
{
    double   dxdz;
    //a_ = b;
    qp_ = qp;

    shapCoeffVD(a_);
    MatrixXd dNdz1;

    dNdz1   = MatrixXd::Zero(a_+1,1);
    for(int i=0;i<=a_-1;i++)
    {
        dNdz1   =  dNdz1 + CoeffdNdz.col(i)* pow(qp_,a_-1-(i));
    }
    dNdz = dNdz1;
    return dNdz;

}





#endif //PRIMALDPG_FEVALUES_H
