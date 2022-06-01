//
// Created by Davood Damirchelli on 2021-10-22.
//

void lgwt(int N, double a, double b)
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

    N  = N-1;
    double N1 = N+1;
    double N2 = N+2;
    VectorXd xu = VectorXd::LinSpaced(N1,-1,1);
    VectorXd p1;
    VectorXd p2;
    VectorXd p3;
    VectorXd one; one.setOnes(N+1,1);
    VectorXd ax;
    MatrixXd L;
    MatrixXd Lp;
    VectorXd y0;
    VectorXd x;
    VectorXd ax2;
    VectorXd w;
    p1 = VectorXd::LinSpaced(N+1,0,N);      // low:step:hi
    p1 = ((2* p1+one)*PI)/(2*N+2);
    p2 = (PI*N/N2)*xu;

    //Initial guess
    p3 = cos(p1.array())+(0.27/N1)*(sin(p2.array()));
    // Legendre-Gauss Vandermonde Matrix
    L  = MatrixXd::Zero(N1,N2);
    y0 = 2*y0.setOnes(N+1,1);
    // Compute the zeros of the N+1 Legendre Polynomial
    // using the recursion relation and the Newton-Raphson method
    double eps = 0.0000000000001;
    int cont;
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


    cout<<x<<endl;
    cout<<"w is "<<endl<<w<<endl;

//return xu;
}

void shapefunction(int n)
{
    cout<<"this is shape function"<<endl;
    VectorXd zc;
    MatrixXd N;
    MatrixXd dNdz;
    MatrixXd d2Ndz2;
    double a;
    zc  = VectorXd::Zero(n+1);
    zc(0) = -1;
    zc(n) = 1;
    for (int i = 1;i<n-1;i++)
    {
        zc(i) = zc(i-1)+(2.0/n);
    }
    MatrixXd z;
    z  = MatrixXd::Zero(n+1,n);

    VectorXd D;
    D  = VectorXd::Zero(n+1);
    for (int k = 1;k<n;k++)
    {
        if (k==1)
        {
            a = 1.0;
            for (int j = 0;j<n-1;j++)
            {
                z(k,j) = zc(j+1);
                a = a*(zc(k)-zc(j+1));
            }
            D(k) = a;
        }
        else if(k>0&&k<n+1)
        {
            a = 1.0;
            for (int j = 0; j < k - 1; j++) {
                a = a * (zc(k) - zc(j));
            }
            for (int j = k - 1; j < n - 1; j++) {
                z(k, j) = zc(j);
                a = a * (zc(k) - zc(j));
            }
            D(k) = a;
        }
        else if(k==n+1)
        {
            a = 1;
            for (int j = k-1;j<n-1;j++)
            {
                z(k,j) = zc(j);
                a = a * (zc(k)-zc(j));
            }
            D(k) = a;
        }






    }
//---------------- now we try to find out the numerator/s

    N =  MatrixXd::Zero(n+1,n+1);
    for(int k=0;k<=n;k++)

    {
        //???????????????
        //N.block(k-1,0,1,n)= POLY(z.block(k-1,0,1,n))/D(k);
        for(int k1=0;k1<n;k1++)
        {
            if (abs(N(k,k1))<0.0000000001)
                N(k,k1) = 0;
        }
    }

    dNdz =  MatrixXd::Zero(n+1,n);

    for(int k=0;k<=n;k++)
    {
        for(int l=0;l<n-1;l++)
        {
            dNdz(k,l) = N(k,l) * (n+1-l);
            if (abs(dNdz(k,l))<0.0000000001)
                dNdz(k,l) = 0;
        }
    }
    if (n==1)
    {
        d2Ndz2 =  MatrixXd::Zero(n+1,1);
    }
    else
    {
        d2Ndz2 =  MatrixXd::Zero(n+1,n-1);

        for(int k=0;k<=n;k++)
        {
            for(int m=0;m<n-2;m++)
            {
                d2Ndz2(k,m)=dNdz(k,m)*(n-m);

                if (abs(d2Ndz2(k,m))<0.0000000001)
                    d2Ndz2(k,m) = 0;
            }
        }

    }




    cout<<zc<<endl;
}

shapeCom shapefunctionManoal(int n)
{


    switch (n)
    {

        case 1:
        {

            shapeCom one;
            //one.zc.assign(-1.0, 1.0);
            //one.zc     <<-1.0, 1.0;
            //cout<<"I am here"<<endl;
            //one.N      <<-0.5,0.5,
            //            0.5,0.5;
            /*
             struct test{

                 VectorXd  zc(2);
                 VectorXd  dNdz(2);
                 VectorXd  d2Ndz2(2);
                 MatrixXd  N(2,2);

                 zc     <<-1.0, 1.0;

                 N      <<-0.5,0.5,
                 0.5,0.5;

                 dNdz   << -0.5, 0.5;
                 d2Ndz2 <<0.0, 0.0;

                 //cout<<endl<<N;
             };
*/
            return one;

            //break;

        }

        case 2:
        {
            VectorXd  zc(3);
            Matrix3d  N;
            MatrixXd  dNdz(3,2);
            VectorXd  d2Ndz2(3);



            zc     << -1.0,0,1.0;
            N      <<-0.5,0.5,0.0,
                    -1.0,0.0,1.0,
                    0.5,0.5,0.0;

            dNdz   << 1.0, -0.5,
                    -2.0,0.0,
                    1.0,0.5;


            d2Ndz2 << 1.0, -2.0,1.0;
            break;
        }

        case 3:
        {
            VectorXd  zc(4);
            Matrix4d  N;
            MatrixXd  dNdz(4,3);
            MatrixXd  d2Ndz2(4,2);



            zc     << -1.0000 ,  -0.3333,    0.3333, 1.0000;
            N      << -0.5625,    0.5625,    0.0625,   -0.0625,
                    1.6875,   -0.5625,   -1.6875,    0.5625,
                    -1.6875,   -0.5625,    1.6875,    0.5625,
                    0.5625,    0.5625,   -0.0625,   -0.0625;

            dNdz   <<  -1.6875,    1.1250,    0.0625,
                    5.0625,   -1.1250,   -1.6875,
                    -5.0625,   -1.1250,   1.6875,
                    1.6875,    1.1250,   -0.0625;


            d2Ndz2 <<  -3.3750,    1.1250,
                    10.1250,   -1.1250,
                    -10.1250,   -1.1250,
                    3.3750 ,   1.1250;
            cout<<endl<<N;
            break;
        }
        case 4:
        {
            VectorXd  zc(5);
            MatrixXd  N(5,5);
            MatrixXd  dNdz(5,4);
            MatrixXd  d2Ndz2(5,3);



            zc     <<  -1.0000,   -0.5000,         0 ,   0.5000 ,   1.0000;
            N      <<  0.6667,   -0.6667,   -0.1667,    0.1667 ,        0,
                    -2.6667 ,   1.3333,    2.6667,   -1.3333 ,        0,
                    4.0000 ,        0,   -5.0000,         0,    1.0000,
                    -2.6667,   -1.3333,    2.6667,    1.3333,         0,
                    0.6667,    0.6667,   -0.1667,   -0.1667,         0;

            dNdz   <<   2.6667,   -2.0000,   -0.3333,    0.1667,
                    -10.6667,    4.0000,    5.3333,   -1.3333,
                    16.0000,         0,  -10.0000,         0,
                    -10.6667,   -4.0000,    5.3333,    1.3333,
                    2.6667,    2.0000,   -0.3333,   -0.1667;


            d2Ndz2 <<   8.0000 ,  -4.0000,   -0.3333,
                    -32.0000,    8.0000,    5.3333,
                    48.0000,         0,  -10.0000,
                    -32.0000,   -8.0000,    5.3333,
                    8.0000,    4.0000,   -0.3333;
            break;
        }
        case 5:
        {
            VectorXd  zc(6);
            MatrixXd  N(6,6);
            MatrixXd  dNdz(6,5);
            MatrixXd  d2Ndz2(6,4);

            zc     << -1.0000,   -0.6000,   -0.2000,    0.2000,    0.6000 ,   1.0000;
            N      <<
                   -0.8138,    0.8138 ,   0.3255 ,  -0.3255,   -0.0117 ,   0.0117,
                    4.0690,   -2.4414,   -4.2318,    2.5391 ,   0.1628,   -0.0977,
                    -8.1380,    1.6276,   11.0677,   -2.2135,   -2.9297,    0.5859,
                    8.1380,    1.6276,  -11.0677,   -2.2135,    2.9297,    0.5859,
                    -4.0690,   -2.4414,    4.2318,    2.5391,   -0.1628,   -0.0977,
                    0.8138,    0.8138,   -0.3255,   -0.3255,    0.0117,    0.0117;

            dNdz   <<-4.0690 ,   3.2552,    0.9766,   -0.6510,   -0.0117,
                    20.3451,   -9.7656 , -12.6953,    5.0781 ,   0.1628,
                    -40.6901,    6.5104 ,  33.2031,   -4.4271,   -2.9297,
                    40.6901 ,   6.5104,  -33.2031,   -4.4271,    2.9297,
                    -20.3451,   -9.7656 ,  12.6953,    5.0781,   -0.1628,
                    4.0690,    3.2552,   -0.9766,   -0.6510,    0.0117;


            d2Ndz2 <<    -16.2760,    9.7656,    1.9531,   -0.6510,
                    81.3802,  -29.2969,  -25.3906,    5.0781,
                    -162.7604,   19.5313,   66.4062,   -4.4271,
                    162.7604,   19.5312,  -66.4062,   -4.4271,
                    -81.3802,  -29.2969,   25.3906,    5.0781,
                    16.2760,    9.7656,   -1.9531,   -0.6510;
            break;
        }
        case 6:
        {
            VectorXd  zc(7);
            MatrixXd  N(7,7);
            MatrixXd  dNdz(7,6);
            MatrixXd  d2Ndz2(7,5);

            zc     <<  -1.0000,   -0.6667,   -0.3333,   -0.0000,    0.3333,    0.6667,    1.0000;
            N      <<  1.0125,   -1.0125,   -0.5625 ,   0.5625 ,   0.0500,   -0.0500 ,        0.0000,
                    -6.0750 ,   4.0500,    6.7500,   -4.5000,   -0.6750,    0.4500,         0.0000,
                    15.1875,   -5.0625,  -21.9375,    7.3125,    6.7500,   -2.2500,         0.0000,
                    -20.2500,         0.0000,   31.5000,         0.0000,  -12.2500 ,        0.0000,    1.0000,
                    15.1875,    5.0625,  -21.9375 ,  -7.3125,    6.7500 ,   2.2500,         0.0000,
                    -6.0750,   -4.0500,    6.7500,    4.5000,   -0.6750 ,  -0.4500,         0.0000,
                    1.0125,    1.0125,   -0.5625,   -0.5625 ,   0.0500 ,   0.0500 ,        0.0000;

            dNdz   <<    6.0750,   -5.0625,   -2.2500,    1.6875,    0.1000,   -0.0500,
                    -36.4500,   20.2500,   27.0000,  -13.5000,   -1.3500,    0.4500,
                    91.1250,  -25.3125,  -87.7500,   21.9375,   13.5000,   -2.2500,
                    -121.5000,         0.00,  126.0000,         0.00,  -24.5000,         0.00,
                    91.1250,   25.3125,  -87.7500,  -21.9375,   13.5000,    2.2500,
                    -36.4500,  -20.2500,   27.0000 ,  13.5000,   -1.3500,   -0.4500,
                    6.0750,    5.0625,   -2.2500,   -1.6875,    0.1000,    0.0500;


            d2Ndz2 <<    30.3750,  -20.2500,   -6.7500,    3.3750,    0.1000,
                    -182.2500,   81.0000,   81.0000,  -27.0000,   -1.3500,
                    455.6250, -101.2500, -263.2500 ,  43.8750,   13.5000,
                    -607.5000,  0.0000,  378.0000,    0.0000,  -24.5000,
                    455.6250,  101.2500, -263.2500,  -43.8750,   13.5000,
                    -182.2500,  -81.0000 ,  81.0000,   27.0000,   -1.3500,
                    30.3750 ,  20.2500 ,  -6.7500,   -3.3750,    0.1000;
            cout<<d2Ndz2;
            break;
        }



    }

}

void quadPointQuantites(double qp)
{
    int NV;
    int NU;
    double   dxdz;
///----- this need to be fixed-------
    shapefunctionManoal(NU);  // cofN ,
    MatrixXd N;
    MatrixXd M;
    MatrixXd dNdz;
    MatrixXd dMdz;
    MatrixXd dNdx;
    MatrixXd dMdx;

    N      = MatrixXd::Zero(NU+1,1);
    dNdz   = MatrixXd::Zero(NU+1,1);

    for(int i=0;i<NU;i++)
    {
        // N    =  N + cofN.col(i-1)* pow(qp,NU-i);
    }

    for(int i=0;i<NU-1;i++)
    {
        // dNdz   =  dNdz + cofN.col(i-1)* pow(qp,NU-1-i);
    }

///----- this need to be fixed-------
    shapefunctionManoal(NV);   //cofM

    M      = MatrixXd::Zero(NV+1,1);
    dMdz   = MatrixXd::Zero(NV+1,1);
    for(int i=0;i<NV;i++)
    {
        // M    =  M + cofM.col(i-1)* pow(qp,NV-i);
    }

    for(int i=0;i<NV-1;i++)
    {
        //dMdz   =  dMdz + cofM.col(i-1)* pow(qp,NV-1-i);
    }

    // dxdz = h/2;
    dNdx       = (1/dxdz ) * dNdz;       // dN/dx
    dMdx       = (1/dxdz)  * dMdz;       // dN/dx
}
/*
void asseblyPrimal()
{
    MatrixXd  B;
    MatrixXd  G;
    MatrixXd  F;


    B = MatrixXd::Zero(NV * n_elems, (NU-1) * n_elems  + 1 +  Nuh * n_elems + 1);
    G = MatrixXd::Zero(NV * n_elems,NV * n_elems);
    F = MatrixXd::Zero(NV * n_elems,1);

    lgwt(quad_pts,-1,1);  ///--------?????????
    decomposeSolution();  ///--------?????????

    for (int iElem = 0;iElem<n_elems-1;iElem++)
    {


        for (int j = 0;j<quad_pts-1;j++)
        {

        }

    }
}
 */