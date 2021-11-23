
/*
 * Example of how to use the mxGPUArray API in a MEX file.  This example shows
 * how to write a MEX function that takes a gpuArray input and returns a
 * gpuArray output, e.g. B=mexFunction(A).
 *
 * Copyright 2012 The MathWorks, Inc.
 */

#include "mex.h"
//#include<stdio.h>
//#include <stdlib.h>
////#include <complex.h>
//#include <math.h>
//#include <omp.h>
//#include <time.h>
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
////#include "cuComplex.h"
//#include "cublas_v2.h"
//#include "gpu/mxGPUArray.h"
//#include <cuComplex.h>
#include <iostream>

#include <arrayfire.h>
#include <cstdio>
#include <cstdlib>
//#include <af/defines.h>
//#include <af/seq.h>





using namespace af;

//
//#define CUDA_CALL(res, str) { if (res != cudaSuccess) { printf("CUDA Error : %s : %s %d : ERR %s\n", str, __FILE__, __LINE__, cudaGetErrorName(res)); } }
//#define CUBLAS_CALL(res, str) { if (res != CUBLAS_STATUS_SUCCESS) { printf("CUBLAS Error : %s : %s %d : ERR %d\n", str, __FILE__, __LINE__, int(res)); } }
//
//
//#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
//inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
//{
//	if (code != cudaSuccess)
//	{
//		char err_str[1000];
//		sprintf(err_str,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//		mexErrMsgTxt(err_str);
//
//	}
//}



/*
 * Device code
 */
//void convec(int FAr, int FAc, int FCr, int FCc)
//{
//
//cublasHandle_t handle ;
//cuDoubleComplex alpha, beta;
//alpha.x = 1;
//alpha.y = 0;
//beta.x = 0;
//beta.y = 0;
//
//int m,n,k,lda,ldb,ldc;
//
////m = FAr;
////n = FAc;
////n = FCr;
////pp = FCc;
//
//
//
//// on the device
//cuDoubleComplex * d_a ;
//// d_a - a on the device
//cuDoubleComplex * d_b ;
//// d_b - b on the device
//cuDoubleComplex * d_c ;
//// d_c - c on the device
//cudaMalloc (( void **)& d_a , FAr * FAc * sizeof (cuDoubleComplex)); // device
//// memory alloc for a
//cudaMalloc (( void **)& d_b , FCr * FCc * sizeof (cuDoubleComplex)); // device
//// memory alloc for b
//cudaMalloc (( void **)& d_c , FAc * FCc * sizeof (cuDoubleComplex)); // device
//// memory alloc for c
//cublasCreate (&handle ); // initialize CUBLAS context
//
//cudaMemcpy(d_a, a, sizeof(cuDoubleComplex)*FAr*FAc, cudaMemcpyHostToDevice);
//cudaMemcpy(d_b, b, sizeof(cuDoubleComplex)*FCr*FCc, cudaMemcpyHostToDevice);
//cublasZgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,FAc,FCc,FAr,&alpha,d_a,FAc,d_b,FAr,&beta,d_c,FAc);
//cudaMemcpy(cc, d_c, sizeof(cuDoubleComplex)*FAc*FCc, cudaMemcpyDeviceToHost);
//
//
//cudaFree ( d_a );
//// free device memory
//cudaFree ( d_b );
//// free device memory
//cudaFree ( d_c );
//// free device memory
//cublasDestroy ( handle );
//// destroy CUBLAS context
//free( a );
//// free host memory
//free( b );
//// free host memory
//free( c );
//
////cuDoubleComplex const * const d_A = d_EB;
////cuDoubleComplex const * const d_B = d_DA;
//
//
////lda = FAc;
////ldb = FCr;
////ldc = FAc;
////m = FAc;
////k = FAr;
////n = FCr;
//
////lda = FAr;
////ldb = FCr;
////ldc = FAr;
////m = FAr;
////k = FAc;
////n = FCc;
//
//
////lda = FAc;
////ldb = FCr;
////ldc = FAc;
////m = FAc;
////k = FAr;
////n = FCc;
//
////if transa == OpN
////m = FAc;
////n = FCr;
////k = FAr;
//
////if transa == OpT
////m = FAc;
////n = FCc;
////k = FAr;
//
////m=2;
////n=2;
////k=3;
//
////lda = FAr;
////ldb = FAc;
////ldc = FAc;
//
////if transa == opN
////lda = m;
////ldb = k;
////ldc = m;
//
////if transa == opT
////lda = 3;
////ldb = 2;
////ldc = 3;
//
//
////
////
////m =  FAr;
////n =  FCc;
////k =  FAc;
////lda = m;
////ldb = k;
////ldc = m;
////
////
////printf(" m : %3d \n",m );
////printf(" n : %3d \n",n );
////printf(" k : %3d \n",k );
////printf(" lda : %3d \n",lda );
////printf(" ldb : %3d \n",ldb );
////printf(" ldc : %3d \n",ldc );
////
////
//////printf(" real dari d_EB[0] : %3d \n",(d_EB[0]) );
////
////
////
//////int m = 2;
//////int n = 3;
//////int pp = 2;
////
////    ///* Calculate the global linear index, assuming a 1-d grid. */
////   // int const i = blockDim.x * blockIdx.x + threadIdx.x;
////    //if (i < N) {
////      //  B[i] = (A[i])+(A[i]);
////    //}
////
////    //B = A*2;
////    double result;
////    cublasStatus_t stat;
////
////    cublasCreate (&handle ); // initialize CUBLAS context
////
////    stat = cublasZgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&alpha,d_EB,lda,d_DA,ldb,&beta,d_C,ldc);
////
////
////
////
////
////
////
////    //printf("ini status : %s", stat);
////    if (stat != CUBLAS_STATUS_SUCCESS) {
////         std::cerr << "***cublasSetMatrix A failed***\n";
////         //return 1;
////     }
////
////
////    //cublasZgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,4,8,6,&alpha,d_A,6,d_C,8,&beta,d_B,4);
////     //cublasZgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,m,pp,n,&alpha,d_A,n,d_C,pp,&beta,d_B,m);
////     //cublasZgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,m,pp,n,&alpha,d_a,n,d_b,pp,&beta,d_c,m);
////     //cublasZgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,3,3,3,&alpha,d_A,3,d_C,3,&beta,d_B,3);
////
////
////  //  stat=cublasDznrm2(handle,m*n,d_B,1,&result);
////    //printf("Euclidean  norm of   x: ");
////   // printf("%7.3f\n",d_A[0] );
////    cublasDestroy ( handle );
//
//}
//
//void __global__ matrixAdd(cuDoubleComplex * const c,cuDoubleComplex b, int const N) {
// int col = blockIdx.x * blockDim.x + threadIdx.x;
// int row = blockIdx.y * blockDim.y + threadIdx.y;
// //int index = row + col * N;
// int index = col + row * N;
// if (col < N && row < N) {
// //cx[index] = c[index];
// if(col == row){
//  c[index].x = c[index].x + b.x;
//  //printf("ini c[index].x : %6.8f \n",c[index].x);
//  }
// }
//}
//
//void __global__ matrixcom(cuDoubleComplex * const c,double * re,double * im, int const fr, int const fc) {
// int col = blockIdx.x * blockDim.x + threadIdx.x;
// int row = blockIdx.y * blockDim.y + threadIdx.y;
// //int index = row + col * N;
// int index = col + row * fr;
// if (col < fr && row < fc) {
// //cx[index] = c[index];
//
//  c[index].x = re[index];
//  c[index].y = im[index];
////
//  //c[index].x = (double)index;
//  //c[index].y = 0;
//  //printf("ini c[index].x : %6.8f \n",c[index].x);
//
// }
//}
//
//
////void cuInversefft_reconed(int nx,int ny,int time,int coil,int slice);
//void pisah(cuDoubleComplex * d_col, int cr, int cc, double * buffercr, double * bufferci);

/*
 * Host code
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/
    //mxGPUArray const *A;
    //mxGPUArray const *C;
    //mxGPUArray const *m;
    //mxGPUArray const *n;
    //mxGPUArray const *pp;
    //mxGPUArray *B;
//    cuDoubleComplex const *d_A;
//    cuDoubleComplex const *d_C;
    int FAr;
    int FAc;
    int FCr;
    int FCc;
    int FTr;
    int FTc;
    //cuDoubleComplex *d_B;

    //dat_mux, ker, us_msk, method,
//    int N;
   // char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
   // char const * const errMsg = "Invalid input to MEX file.";

    /* Choose a reasonably sized number of threads for the block. */
//    int const threadsPerBlock = 256;
//    int blocksPerGrid;

    /* Initialize the MathWorks GPU API. */
    //mxInitGPU();

    /* Throw an error if the input is not a GPU array. */
//    if ((nrhs!=1) || !(mxIsGPUArray(prhs[0]))) {
   // printf("jumlah : ");
//        mexErrMsgIdAndTxt(errId, errMsg);
//    }
const size_t *dim_arrayA1,*dim_arrayB1,*dim_arrayA2,*dim_arrayB2,*dim_arrayC,*dim_arrayD,*dim_arrayT;
int number_of_dims, c;
number_of_dims = mxGetNumberOfDimensions(prhs[0]);
//printf("number_of_dims : %1.2f   \n",number_of_dims);
    dim_arrayA1 = mxGetDimensions(prhs[0]);
    dim_arrayB1 = mxGetDimensions(prhs[2]);
    dim_arrayA2 = mxGetDimensions(prhs[3]);
    dim_arrayB2 = mxGetDimensions(prhs[4]);
//    dim_arrayC = mxGetDimensions(prhs[4]);
//    dim_arrayD = mxGetDimensions(prhs[5]);
    //dim_arrayC = mxGetDimensions(prhs[1]);
    //dim_arrayT = mxGetDimensions(prhs[2]);
   // for (c=0; c<number_of_dims; c++)
     //  mexPrintf("%d\n", (dim_arrayA[0]*dim_arrayC[1]));
double *Ar1=(double*)mxGetPr(prhs[0]);
double *Ai1=(double*)mxGetPi(prhs[0]);

double *Kr=(double*)mxGetPr(prhs[1]);
//double *Ki=(double*)mxGetPi(prhs[1]);

double *Br1=(double*)mxGetPr(prhs[2]);
double *Bi1=(double*)mxGetPi(prhs[2]);

double *Ar2=(double*)mxGetPr(prhs[3]);
double *Ai2=(double*)mxGetPi(prhs[3]);

double *Br2=(double*)mxGetPr(prhs[4]);
double *Bi2=(double*)mxGetPi(prhs[4]);

//double *nmskr=(double*)mxGetPr(prhs[2]);
//double *nsampr=(double*)mxGetPr(prhs[3]);
//
//double *nomgr=(double*)mxGetPr(prhs[4]);
//double *nomgi=(double*)mxGetPi(prhs[4]);
//
//double *nkzr=(double*)mxGetPr(prhs[5]);
//double *nkzi=(double*)mxGetPi(prhs[5]);

//double *Cr=(double*)mxGetPr(prhs[1]);
//double *Ci=(double*)mxGetPi(prhs[1]);

//double *Tr=(double*)mxGetPr(prhs[2]);
//double *Ti=(double*)mxGetPi(prhs[2]);
    int FA11,FA21,FA31,FA41,K11,K21,K31,K41,FA12,FA22,FA32,FA42,K12,K22,K32,K42;
    //FA4 = 0;
    FA11 = dim_arrayA1[0];
    FA21 = dim_arrayA1[1];
    FA31 = dim_arrayA1[2];
    FA41 = dim_arrayA1[3];
    FA12 = dim_arrayA2[0];
    FA22 = dim_arrayA2[1];
    FA32 = dim_arrayA2[2];
    FA42 = dim_arrayA2[3];

    K11 = dim_arrayB1[0];
    K21 = dim_arrayB1[1];
    K31 = dim_arrayB1[2];
    K41 = dim_arrayB1[3];
    K12 = dim_arrayB2[0];
    K22 = dim_arrayB2[1];
    K32 = dim_arrayB2[2];
    K42 = dim_arrayB2[3];

//    O1 = dim_arrayC[0];
//    O2 = dim_arrayC[1];
//    KZ1 = dim_arrayD[0];
//    KZ2 = dim_arrayD[1];
    //K5 = dim_arrayB[4];
    //printf("FAr : %d   \n",FAr);

    //std::cout << "FA1 = "<< FA1<< std::endl;
//    std::cout << "FA2 = "<< FA2<< std::endl;
//    std::cout << "FA3 = "<< FA3<< std::endl;
//    std::cout << "FA4 = "<< FA4<< std::endl;



//    FCr = dim_arrayC[0];
//    FCc = dim_arrayC[1];
//    FTr = dim_arrayT[0];
//    FTc = dim_arrayT[1];

//
//cuDoubleComplex *a ;
//// mxk matrix a on the host
//cuDoubleComplex *b ;
// kxn matrix b on the host
//cuDoubleComplex *cc ;
//// kxn matrix b on the host
//cuDoubleComplex *ct ;
//// kxn matrix b on the host
//cuDoubleComplex *cct ;
//// mxn matrix c on the host
////a =( cuDoubleComplex  *) malloc (( FAr*FAc) * sizeof (cuDoubleComplex));
////// host memory for a
////b =(cuDoubleComplex *) malloc (( FCr*FCc) * sizeof ( cuDoubleComplex ));
//// host memory for b
//cc =( cuDoubleComplex *) malloc (( FCc*FCc) * sizeof ( cuDoubleComplex ));
//// host memory for b
//cct =( cuDoubleComplex *) malloc (( FCc*FAc) * sizeof ( cuDoubleComplex ));
//
//// host memory for t
////ct =( cuComplex *) malloc (( FTc*FTc) * sizeof ( cuComplex ));
////conjugate transpose of A :
////for(int i=0;i<FAr;i++){
// //for (int j=0;j<FAc;j++){
//	//a[i*FAc+j].x= Ar[i+FAr*j];
//	//a[i*FAc+j].y= -1*Ai[i+FAr*j];
//    //}
//    //}
////
////for(int i=0;i<FCc;i++){
//// for (int j=0;j<FCr;j++){
////	b[i*FCr+j].x= Cr[i*FCr+j];
////	b[i*FCr+j].y= Ci[i*FCr+j];
////    }
////    }
////
////for(int i=0;i<FAc;i++){
//// for (int j=0;j<FAr;j++){
////	a[i*FAr+j].x= Ar[i*FAr+j];
////	a[i*FAr+j].y= Ai[i*FAr+j];
////    }
////    }
//
//
//
//
//    cublasHandle_t handle ;
//cuDoubleComplex alpha, beta;
//alpha.x = 1;
//alpha.y = 0;
//beta.x = 0;
//beta.y = 0;
//
//int m,n,k,lda,ldb,ldc;
//
//
//
//
////m = FAr;
////n = FAc;
////n = FCr;
////pp = FCc;
//double result;
////cublasStatus_t stat;
//
//// on the device
//cuDoubleComplex * d_a ;
//// on the device
//double * d_ar ;
//// on the device
//double * d_ai ;
//// d_a - a on the device
//cuDoubleComplex * d_b ;
//// d_a - a on the device
//double * d_br ;
//// d_a - a on the device
//double * d_bi ;
//// d_b - b on the device
//cuDoubleComplex * d_c ;
//// d_c - c on the device
//cuDoubleComplex * d_ct ;
//
//



try {



//
//        // Select a device and display arrayfire info
//
            //af::info();
            //FA4 =1;
//            if(number_of_dims==3){
//            FA4 = 1;
//            }



            double *xpanr;
            double *xpani;
            int pnj;

            int p4t = FA41 + FA42;
for (int ik=0;ik<2;ik=ik+1)
{

if(ik==1){

FA11=FA12;
FA21=FA22;
FA31=FA32;
FA41=FA42;
Ar1 = Ar2;
Ai1 = Ai2;

K11=K12;
K21=K22;
K31=K32;
K41=K42;
Br1=Br2;
Bi1=Bi2;


}
//            array dat_muxr;
//            array dat_muxi;
//            array kerr;
//            array keri;
            array dat_muxr(FA11,FA21,FA31,FA41,Ar1);
            array dat_muxi(FA11,FA21,FA31,FA41,Ai1);
            array kerr(K11,K21,K31,K41,Br1);
            array keri(K11,K21,K31,K41,Bi1);



//if(ik==0){
//            dat_muxr(FA11,FA21,FA31,FA41,Ar1);
//            dat_muxi(FA11,FA21,FA31,FA41,Ai1);
//            kerr(K11,K21,K31,K41,Br1);
//            keri(K11,K21,K31,K41,Bi1);
//            }
//
//if(ik==1){
//            dat_muxr(FA12,FA22,FA32,FA42,Ar2);
//            dat_muxi(FA12,FA22,FA32,FA42,Ai2);
//            kerr(K12,K22,K32,K42,Br2);
//            keri(K12,K22,K32,K42,Bi2);
//            }
//            array nmsk(1,nmskr);
//            array nsamp(1,nsampr);
//
//            array omgr(O1,O2,1,1,nomgr);
//            //array omgi(O1,O2,1,1,nomgi);
//            array kzr(KZ1,KZ2,nkzr);
            //array kzi(KZ1,KZ2,nkzi);
            //af_print(kzr);
            //printf("numdims(a)  %d\n", A.numdims()); // 3
            int x1,y1,z1,w1;
            int p1,p2,p3,p4,row,nz;
            if(ik==0){
            p1 = FA11;
            p2 = FA21;
            p3 = FA31;
            p4 = FA41;}

            if(ik==1){
            p1 = FA12;
            p2 = FA22;
            p3 = FA32;
            p4 = FA42;}

//            nz = 6;
//            row = p3*nz;
//            row = K4;

            array A = complex(dat_muxr,dat_muxi);
            array B = complex(kerr,keri);

//            std::cout << "A dim 1 = "<< A.dims(0) << std::endl;
//            std::cout << "A dim 2 = "<< A.dims(1) << std::endl;
//            std::cout << "A dim 3 = "<< A.dims(2) << std::endl;
//            std::cout << "A dim 4 = "<< A.dims(3) << std::endl;

            int a1 = *Kr;

            x1=0;
            y1=0;
            z1=0;
            w1=0;

            if(a1==1){
            x1 =1*floor(p1/2);}
            if(a1==2){
            y1 =1*floor(p2/2);}
            if(a1==3){
            z1 =1*floor(p3/2);}
            if(a1==3){
            w1 =1*floor(p4/2);}


            array Bimag;
            array Breal;

            //int pnjt = 0;
            double *h_ptr;
            double *h_pti;

            int p4s = p4;
            int pem = 6;
            int p4x = floor(p4/pem);
            int p4as = p4;
            int p4a = p4x;
//            std::cout << "aaaaa "<< p1 << std::endl;
//            std::cout << "aaaaa "<< p2 << std::endl;
//            std::cout << "aaaaa "<< p3 << std::endl;
//            std::cout << "aaaaa "<< p4 << std::endl;
//            std::cout << "aaaaa "<< p4x << std::endl;
            //A(span,seq(1,end,2),span,span) = -1*A(span,seq(1,end,2),span,span);
            //std::cout << "aaaaa "<< p4 << std::endl;
            array As = A;
            array Bs = B;

//            std::cout << "As dim 1 = "<< As.dims(0) << std::endl;
//            std::cout << "As dim 2 = "<< As.dims(1) << std::endl;
//            std::cout << "As dim 3 = "<< As.dims(2) << std::endl;
//            std::cout << "As dim 4 = "<< As.dims(3) << std::endl;

            for(int k=0;k<pem;k = k + 1){
           //std::cout << "K = "<< k << std::endl;

           if(k==(pem-1)){
           p4x=p4s-(p4x*(pem-1));
           //hp = constant(0,p1,p2,p4x,row,c64);

           }

           //std::cout << "K lagi = "<< k << std::endl;
          gfor(seq i, p4x){
            A = As(span,span,span,i+p4a*k);
            B = Bs(span,span,span,i+p4a*k);
            }
//          gfor(seq i, p4x){
//            B = Bs(span,span,span,i+p4a*k);}

            ///ksp(:, :, echo, :, :,t) = fftc( ifftc(ksp(:, :, echo, :, :,t), p.FE_DIM) .* p.pha_flt, p.FE_DIM); % The same coefficients are used for every echo and every time point.
//            std::cout << "A dim 1 = "<< A.dims(0) << std::endl;
//            std::cout << "A dim 2 = "<< A.dims(1) << std::endl;
//            std::cout << "A dim 3 = "<< A.dims(2) << std::endl;
//            std::cout << "A dim 4 = "<< A.dims(3) << std::endl;
//            gfor(seq i, p4x){
//            A = A(span,span,span,i+p4a*k);}
//            gfor(seq i, p4x){
//            B = B(span,span,span,i+p4a*k);}

//            if(k==(pem-1)){
//            array Ax = A(span,span,span,seq(p4a*k,end));
//            array Bx = B(span,span,span,seq(p4a*k,end));
//            }else{
//            array Ax = A(span,span,span,seq(p4a*k,p4a*(k+1)));
//            array Bx = B(span,span,span,seq(p4a*k,p4a*(k+1)));
//            }
//
//            std::cout << "A dim akhir = "<< Ax.dims(3) << std::endl;
//            std::cout << "B dim akhir = "<< Bx.dims(3) << std::endl;
//            for(int i=0; i<p4x; i=i+1){
//            A = A(span,span,span,i+p4a*k);
//            B = B(span,span,span,i+p4a*k);}

            //std::cout << "aaaaa "<< 6 << std::endl;
            A(span,seq(1,end,2),span,span) = -1*A(span,seq(1,end,2),span,span);

            //std::cout << "aaaaa "<< 2 << std::endl;

            //gfor(seq i, p4x){
           // A = shift(A(span,span,span,i+p4a*k),x1,y1,z1,w1);}

            A = shift(A,x1,y1,z1,w1);
            //std::cout << "aaaaa "<< 3 << std::endl;
            //gfor(seq i, p4x){
            //for(int i=0; i<p4x; i=i+1){
            //B = B(span,span,span,i+p4a*k);}

            //A = shift(A,x1,y1,z1,w1);
            A = (sqrt(A.dims(0)*A.dims(1)))*ifft(A);
            A = shift(A,-x1,-y1,-z1,-w1);
            A = A*B;
            A = shift(A,x1,y1,z1,w1);
            A = (1/sqrt(A.dims(0)*A.dims(1)))*fft(A);
            A = shift(A,-x1,-y1,-z1,-w1);
            A = moddims(A,1,p1*p2*p3*p4x);


            int ndim    = 2;
                Bimag = imag(A);
                Breal = real(A);

            if(k==0 && ik==0){

             const size_t dims5[2] = { 1,p1*p2*p3*p4t};
             plhs[0] = mxCreateNumericArray( ndim,dims5,mxDOUBLE_CLASS,mxCOMPLEX);

             h_ptr = (double *)mxGetPr(plhs[0]);
             //std::cout << "dim B4 = "<< B.dims(0) << std::endl;
             h_pti = (double *)mxGetPi(plhs[0]);


                Breal.host(h_ptr);
                Bimag.host(h_pti);
                xpanr = h_ptr;
                xpani = h_pti;
             }else{
             Breal.host(xpanr+pnj);
             xpanr = xpanr+pnj;
             Bimag.host(xpani+pnj);
             xpani = xpani+pnj;

             }

             pnj = p1*p2*p3*p4x;

             }





}





























//             const size_t dims5[4] = {FA1,FA2,FA3,FA4};
//             plhs[0] = mxCreateNumericArray( ndim,dims5,mxDOUBLE_CLASS,mxCOMPLEX);
//
//             double *h_ptr = (double *)mxGetPr(plhs[0]);
//             //std::cout << "dim B4 = "<< B.dims(0) << std::endl;
//             double *h_pti = (double *)mxGetPi(plhs[0]);
//
//
//                Breal.host(h_ptr);
//                Bimag.host(h_pti);

//            //array F = constant(1,1,row,c64);
//
////            A = shift(A,x1,y1,z1,w1);
////
////
////            A = (sqrt(A.dims(0)*A.dims(1)))*ifft2(A);
////            A = shift(A,-x1,-y1,z1,w1);
////            //array d = A;
////
////            //af_print(A(1));
////            A = moddims(A,p1,p2,p3,p4);
//            //A = matmul(A, F);
//
////            K = moddims(K,p1,p2,p3,row);
//            //K = K.T();
//            //K = moddims(K,K.dims(0)*K.dims(1),1);
//            //array F1 = constant(1,1,p4,c64);
//            //K = matmul(K, F1);
//            //K = moddims(K,row,p4*p1*p2*p3);
//            //K = K.T();
//            //array s ;
//
//
////            array Ax = constant(1,5,5,5);
////            array Bx = constant(1,5,5);
////            gfor (seq k, 5)
////            Ax(span,span,k) = (k+1)*Bx + sin(k+1);  // expressions
////            af_print(Ax);
//
//            //std::cout << "ini A dims yg prtma = "<< A.dims(0) << std::endl;
//
//            //array h = constant(0,p1,p2,p4,row,c64);
//            array Bx;
//            array h;
//            array hx;
//            array As = A;
//            array Ks = K;
//            int p4s = p4;
//            int pem = 6;
//            int p4x = floor(p4/pem);
//            int p4as = p4;
//            int p4a = p4x;
//            array hp = constant(0,p1,p2,p4x,row,c64);
//            array hasilr;
//
//
//            int ndim    = 2;         /* Number of dimensions */
//            const size_t dims[4] = { p1,p2,p4x, p3*nz }; /* Size of dimensions   */
//            //plhs[k] = mxCreateNumericArray( ndim,dims,mxDOUBLE_CLASS,mxCOMPLEX);
//            double *buffercr; //= (double *)mxGetPr(plhs[k]);
//            //std::cout << "dim B4 = "<< B.dims(0) << std::endl;
//            double *bufferci; //= (double *)mxGetPi(plhs[k]);
//            int pea = nz;
//            int rapx;
//            int rax1;
//            double *xpanr;
//            double *xpani;
//            int pnj;
//            int pnjt = 0;
//            double *h_ptr;
//            double *h_pti;
//
//
//            //array hxx = constant(0,p1,p2,p4,row);
//            //p4x = 30;
//           for(int k=0;k<pem;k = k + 1){
//           //std::cout << "K = "<< k << std::endl;
//
//           if(k==(pem-1)){
//           p4x=p4s-(p4x*(pem-1));
//           hp = constant(0,p1,p2,p4x,row,c64);
//
//           }
//
//           //std::cout << "K lagi = "<< k << std::endl;
//           A = As;
//           K = Ks;
//            //std::cout << "K pertama sblum = "<< k << std::endl;
//           gfor(seq i, p4x){
//            A = shift(A(span,span,span,i+p4a*k),x1,y1,z1,w1);}
//            //std::cout << "K pertama sesudah = "<< k << std::endl;
//
//
//            A = (sqrt(A.dims(0)*A.dims(1)))*ifft2(A);
//            A = shift(A,-x1,-y1,z1,w1);
//            //array d = A;
//
//            //af_print(A(1));
//            A = moddims(A,p1,p2,p3,p4x);
//
//
//
//
//           for(int i=0;i<p4x;i = i + 1){
//            //std::cout << "dim4 after fft2 kdua = "<< 5555 << std::endl;
//                gfor(seq j, row){
//                   hx = sum((K(span,span,span,j) * A(span,span,span,i)),2);
//                   //hxx(span,span,j) = moddims(hx,p1,p2,1);
//                //af_print(h.dims());
////                std::cout << "nsamp1 = "<< h.dims(0) << std::endl;
////                std::cout << "nsamp2 = "<< h.dims(1) << std::endl;
////                std::cout << "nsamp3 = "<< h.dims(2) << std::endl;
////                std::cout << "nsamp4 = "<< h.dims(3) << std::endl;
////                std::cout << "j = "<< i << std::endl;
////                af_print(j);
//                }
//               // hx = moddims(hx,p1,p2,row,1);
////                std::cout << "nsamp1 = "<< hx.dims(0) << std::endl;
////                std::cout << "nsamp2 = "<< hx.dims(1) << std::endl;
////                std::cout << "nsamp3 = "<< hx.dims(2) << std::endl;
////                std::cout << "nsamp4 = "<< hx.dims(3) << std::endl;
////                //std::cout << "j = "<< i << std::endl;
//                hp(span,span,i,span) = hx(span,span,span,span);
////                std::cout << "nsamp1 = "<< h.dims(0) << std::endl;
////                std::cout << "nsamp2 = "<< h.dims(1) << std::endl;
////                std::cout << "nsamp3 = "<< h.dims(2) << std::endl;
////                std::cout << "nsamp4 = "<< h.dims(3) << std::endl;
//
//
//                }
////                if(k==0){
////                h = hp;}else{
////                h = join(2,h,hp);}
////                eval(h);
////                eval(hx);
////                eval(hp);
////                sync();
//
//
//                //h = join(1, h, moddims(hp,1,p1*p2*p4x*row));
//                //h(span,span,i,span) = hp(span,span,span,span);
//
////                eval(h);
////                eval(hx);
//////                h.eval();
//////                sync();
////
////                }
//
//
////                std::cout << "nsamp1 = "<< h.dims(0) << std::endl;
////                std::cout << "nsamp2 = "<< h.dims(1) << std::endl;
////                std::cout << "nsamp3 = "<< h.dims(2) << std::endl;
////                std::cout << "nsamp4 = "<< h.dims(3) << std::endl;
//                //for(int i=0; i< p4; i++){
////                    h(span,span,0,j)= sum((K(span,span,span,0) * A(span,span,span,0)),2);
//                   // }
//
//                    //h(span,span,i,j) = sum(s, 2);
//            //h.eval();
//            //sync();
//            //A = A * K;
//            //A = moddims(A,p1,p2,p3,p4*row);
//            //A = sum(A, 2);
//            p4 = p4x;
//
//            A = shift(hp,x1,y1,z1,w1);
//            //std::cout << "dim A = "<< A.dims(0) << std::endl;
//            //gfor(seq j, row)
//            //for(int j=0;j<p4;j = j + 1){
//                //A(span,span,j,span) = (1/sqrt(A.dims(0)*A.dims(1)))*fft2(A(span,span,j,span));}
//                //std::cout << "K kedua sebelum = "<< k << std::endl;
//            //for(int i=0;i<row;i = i + 1){
//            //A(span,span,span,i) = (1/sqrt(A.dims(0)*A.dims(1)))*fft2(A(span,span,span,i));}
//            //A = (1/sqrt(A.dims(0)*A.dims(1)))*fft2(A);
//            rapx = floor(row/pea);
//            rax1 = rapx;
//
//            for(int i=0;i<pea;i = i + 1){
//            if(i==(pea-1)){rax1 = row - rapx*i;}
//            gfor(seq j, rax1){
//            A(span,span,span,j+(rapx)*i) = (1/sqrt(A.dims(0)*A.dims(1)))*fft2(A(span,span,span,j+(rapx)*i));}
//
//            }
//
//            //std::cout << "K kedua sesudah = "<< k << std::endl;
//
//            //std::cout << "dim A = "<< A.dims(0) << std::endl;
//            A = shift(A,x1,y1,z1,w1);
//            A = moddims(A,p1*p2*p4,row);
//
//            //std::cout << "dim4 after fft2 kdua = "<< A.dims(1) << std::endl;
//
//            //array d = A;
//
//            ///area encoded_ftz_pha :
//            // int nmsk =1;
//             //int nsamp = 4;
//             //array mfp = constant(0,nsamp,nz,nmsk,c64);
//             int a1 = 2*floor(nz/2);
//             int a2 = *nmskr * *nsampr;
//             //array b = iota(dim4(a1, 3), dim4(1, 2));
//
//             array z = iota(dim4(1, a1),dim4(a2,1))-floor(nz/2);
////             array mok = randu(nsamp,1,nmsk,c64); ///90*1*nmsk omegaz
////             array msk = floor(randu(nsamp*nmsk,1,f32)*nsamp); ///msk.kz index utk milih nilai mok
//              array mok = moddims(omgr(kzr-1),a2,1);
//             mok = matmul(mok,constant(1,1,nz,f64));
////             array q = constant(1,nsamp*nmsk,nz,f64);
//             mok = (mok * z)*complex(0,constant(1,a2,nz,f64));
//             mok = exp(mok); ///90*6*nmsk
//
//             //std::cout << "dim mok = "<< mok.dims(0) << std::endl;
//
//             array B = conjg(matmul(constant(1,p1,1,c64),moddims((moddims(matmul(constant(1,p4*p3,1,c64),moddims(mok.T(),1,p2*nz)),p4*p3*nz,p2)).T(),1,p4*p3*nz*p2)));
//             //std::cout << "dim B = "<< B.dims(0) << std::endl;
//             B = moddims(B, p1*p2*p4, p3*nz);
//             //std::cout << "dim B1 = "<< B.dims(0) << std::endl;
//             B = A * B;
//             //std::cout << "dim B2 = "<< B.dims(0) << std::endl;
//             B = moddims(B, p1*p2,p4, p3,nz);
//             B = reorder(B,0,3,2,1);
//             B = moddims(B,1,p1*p2*nz*p3*p4);
//
//             array Bimag = imag(B);
//             array Breal = real(B);
//             if(k==0){
//
//             const size_t dims5[2] = { 1,p1*p2*p3*nz*p4as};
//             plhs[0] = mxCreateNumericArray( ndim,dims5,mxDOUBLE_CLASS,mxCOMPLEX);
//
//             h_ptr = (double *)mxGetPr(plhs[0]);
//             //std::cout << "dim B4 = "<< B.dims(0) << std::endl;
//             h_pti = (double *)mxGetPi(plhs[0]);
//
//
//                Breal.host(h_ptr);
//                Bimag.host(h_pti);
//                xpanr = h_ptr;
//                xpani = h_pti;
//             }else{
//             Breal.host(xpanr+pnj);
//             xpanr = xpanr+pnj;
//             Bimag.host(xpani+pnj);
//             xpani = xpani+pnj;
//
//             }
//                pnj = p1*p2*nz*p3*p4;
//                pnjt = pnjt+pnj;
//
//
////             if(k>0){
////             Bx = join(3,Bx, B);
////             }else{
////             Bx = B;
////
////             }
//             //std::cout << "dim B3 = "<< B.dims(0) << std::endl;
//            //array B = A;
//
//             //af_print(B);
//
////             if(k==0){
////                hasilr = Breal;
////
////                }
////            else{
////                hasilr = join(2,hasilr,Breal);
////                }
//
////             array d;
////             array dimag = imag(d);
////             array dreal = real(d);
//             //af_print(Breal);
////             double * host_a = Breal.host<double>();
////            std::cout << "Breal  = "<< host_a << std::endl;
////             double * datar;
////             double * datai;
//
//             //float * host_a = a.host<float>();
//             //std::cout << "datar  = "<< datar << std::endl;
//
//              //plhs[0] = mxCreateDoubleMatrix(p1,p2,p4, p3*nz, mxCOMPLEX);
////              int ndim    = 4;         /* Number of dimensions */
////              const size_t dims[4] = { p1,p2,p4, p3*nz }; /* Size of dimensions   */
////              plhs[k] = mxCreateNumericArray( ndim,dims,mxDOUBLE_CLASS,mxCOMPLEX);
////            double *buffercr = (double *)mxGetPr(plhs[k]);
////            std::cout << "dim B4 = "<< B.dims(0) << std::endl;
////            double *bufferci = (double *)mxGetPi(plhs[k]);
//
//            //std::cout << "dim B5 = "<< B.dims(0) << std::endl;
//
//
//
////            double *buffercrb[p1][p2][p4][p3*nz];
////            double *buffercib[p1][p2][p4][p3*nz];
////            int ndimm    = 2;         /* Number of dimensions */
////              const size_t dimsm[2] = { p1*p2,row }; /* Size of dimensions   */
////              plhs[1] = mxCreateNumericArray( ndimm,dimsm,mxDOUBLE_CLASS,mxCOMPLEX);
////            double *buffercrm = (double *)mxGetPr(plhs[1]);
////            double *buffercim = (double *)mxGetPi(plhs[1]);
////
////            dreal.host(buffercrm);
////            //free(buffercr);
////            dimag.host(buffercim);
//
////            buffercr = Breal.host<double>();
////            bufferci = Bimag.host<double>();
//            //ty *h_ptr = A.host<ty>();
//            if(k==(pem-1)){
//
//
//
////            const size_t dims5[2] = { 1,p1*p2*p3*nz*p4as};
////            plhs[0] = mxCreateNumericArray( ndim,dims5,mxDOUBLE_CLASS,mxCOMPLEX);
////
////            h_ptr = (double *)mxGetPr(plhs[0]);
////            //std::cout << "dim B4 = "<< B.dims(0) << std::endl;
////            h_pti = (double *)mxGetPi(plhs[0]);
//
//
//           //std::cout << "buffercr = "<< FA4 << std::endl;
//            //std::cout << "bufferci = "<< bufferci << std::endl;
////            if(k==0){
//            //Breal.host(buffercr);
//            //free(buffercr);
//            //Bimag.host(bufferci);
//
//            }
////            }else{
////
////            Breal.host(buffercr+p1*p2*p4*p3*nz*k);
////            //free(buffercr);
////            Bimag.host(bufferci+p1*p2*p4*p3*nz*k);
////
////            }
//            //std::cout << "dim B6 = "<< B.dims(0) << std::endl;
//
//            }

//            eval(Bimag);
//            eval(Breal);
//            sync();
            //free(bufferci);
            //A.host(h_ptr);
//            buffercr = datar;
//            bufferci = datai;
            //memcpy(buffercr, datar, p1*p2*p4*p3*nz*sizeof(double));

            //cudaMemcpy(buffercr, datar, p1*p2*p4*p3*nz*sizeof(double), cudaMemcpyDeviceToHost);
//            cudaMemcpy(bufferci, datai, p1*p2*p4*p3*nz*sizeof(double), cudaMemcpyDeviceToHost);



            //plhs[0] = B.host<double>;       /* declare pointer to mxArray */

            //mlfAssign(&A, mxCreateNumericArray( ndim,
                                    //dims,
                                    //mxDOUBLE_CLASS,
                                    //mxCOMPLEX));






            //af_print(dat_mux);
            //A = moddims(A,p1,p2,p3,p4);











            //af_print(B);
            //af_print(mok);
           // af_print(omgr);

            //std::cout << "nmsk*nsamp = "<< datar << std::endl;
            //std::cout << "nmsk = "<< nmsk << std::endl;
            //std::cout << "dat_mux type = "<< dat_mux.type()<< std::endl;
            //std::cout << "A.numdims() = "<< nsamp.numdims()<< std::endl;
            //std::cout << "nsamp = "<< nsamp << std::endl;
            //std::cout << "B dimensional = "<< B.numdims()<< std::endl;



//
//        int x1,y1,z1,w1;
//        int p1,p2,p3,p4,row,nz;
////        p1 = 90;
////        p2 = 90;
////        p3 = 32;
////        p4 = 368;
////        row = 192;
//
////        array x = iota(dim4(1,16))+1;
////        af_print(x);
////        x = moddims(x,2,2,2,2);
////
////        array y = iota(dim4(1,64))+1;
////        af_print(y);
////dat_mux= reshape(x,2,2,2,2);
////y= 1:64;
////ker= reshape(y,2,2,2,2,2,2);
//
//        p1 = 4;
//        p2 = 4;
//        p3 = 8;
//        p4 = 10;
//        nz = 6;
//        row = p3*nz;
//
//         p1 = 2;
//         p2 = 2;
//         p3 = 2;
//         p4 = 2;
//         nz = 2;
//         row = p3*nz;
//
//        printf("Create a 5-by-3 matrix of random floats on the GPU\n");
////        x= 1:16;
////dat_mux= reshape(x,2,2,2,2);
////y= 1:32;
////ker= reshape(y,2,2,2,2,2)
//
//        array wx = iota(dim4(1,16),dim4(1,1),f64)+1;
//        wx = complex(wx,0);
//        array wy = iota(dim4(1,32),dim4(1,1),f64)+1;
//        wy = complex(wy,0);
//        array A = wx;
//        array K = wy;// ini adalah ker (90,90,32,192)
//
//        A = moddims(A,p1,p2,p3,p4);
//        K = moddims(K,p1,p2,p3,row);
//
//
////        array A = randu(p1,p2,p3,p4,c64);
////        array K = randu(p1,p2,p3,row,c64);// ini adalah ker (90,90,32,192)
//
//        x1 =-1*floor(p1/2);
//        y1 =-1*floor(p1/2);
//        z1 =0;
//        w1 =0;
////        array B = shift(A,x1,y1,z1,w1);
////         array C = fft2(B);
////         array D = shift(C,x1,y1,z1,w1);
////         array E = moddims(D,p1*p2*p3*p4,1);
//         array F = constant(1,1,row,c64);
//
////         array G = matmul(E, F);
//        printf("copy F 4*4*8*16 sebanyak 10 \n");
//        //af_print(G);
//        af_print(A);
//        A = shift(A,x1,y1,z1,w1);
//        A = (1/sqrt(A.dims(0)*A.dims(1)))*fft2(A);
//        A = shift(A,x1,y1,z1,w1);
//        af_print(A);
//        A = moddims(A,p1*p2*p3*p4,1);
//        A = matmul(A, F);
//
//        K = moddims(K,p1*p2*p3,row);
//        K = K.T();
//        K = moddims(K,K.dims(0)*K.dims(1),1);
//        array F1 = constant(1,1,p4,c64);
//        K = matmul(K, F1);
//        K = moddims(K,row,p4*p1*p2*p3);
//        K = K.T();
//
//        af_print(A);
//        A = A * K;
//        af_print(A);
//        A = moddims(A,p1,p2,p3,p4*row);
//        A = sum(A, 2);
////        A = moddims(A,p1*p2*p4,row);
////        af_print(A);
////        A = moddims(A,p1,p2,p4*row);
//        A = shift(A,x1,y1,z1,w1);
//        A = (sqrt(A.dims(0)*A.dims(1)))*ifft2(A);
//        A = shift(A,x1,y1,z1,w1);
//        A = moddims(A,p1*p2*p4,row);
//        af_print(A);
//
//        ///area encoded_ftz_pha :
//         int nmsk =1;
//         int nsamp = 4;
//         array mfp = constant(0,nsamp,nz,nmsk,c64);
//         array z = iota(dim4(1, 2*floor(nz/2)),dim4(nmsk*nsamp,1))-floor(nz/2);
//         array mok = randu(nsamp,1,nmsk,c64); ///90*1*nmsk omegaz
//         array msk = floor(randu(nsamp*nmsk,1,f32)*nsamp); ///msk.kz index utk milih nilai mok
//         mok = moddims(mok,nsamp*nmsk,1);
//         mok = matmul(mok(msk),constant(1,1,nz,c64));
//         array q = constant(1,nsamp*nmsk,nz,f64);
//         mok = (mok * z)*complex(0,q);
//         mok = exp(mok); ///90*6*nmsk
//
////         nx = 2;
////         nc = 2;
////         x= 1:4;
////         us_msk.ftz_pha =  reshape(x,2,2);
//////////////////////////////////////////////////////////////////////////
////         p1 = 2;
////         p2 = 2;
////         p3 = 2;
////         p4 = 2;
////         nz = 2;
////         row = p3*nz;
//         array zx = iota(dim4(1,4),dim4(1,1),f64)+1;
//         zx = complex(zx,0);
//
//         mok = moddims(zx,2,2);
//         af_print(mok);
//
///////////////////////////////////////////////////////////////////////////
//         ///mengenai dat = dat .* conj(repmat(permute(us_msk.ftz_pha, [4,1,3,2]), [nx,1,nc,1]));
//         //array F3 = constant(1,p3,1,c64);
//         array B = conjg(matmul(constant(1,p1,1,c64),moddims((moddims(matmul(constant(1,p4*p3,1,c64),moddims(mok.T(),1,p2*nz)),p4*p3*nz,p2)).T(),1,p4*p3*nz*p2)));
//         //array B = conjg(matmul(constant(1,p4,1,c64),moddims(matmul(constant(1,p1,1,c64),moddims((moddims(matmul(constant(1,p3,1,c64),moddims(mok.T(),1,p2*nz)),p3*nz,p2)).T(),1,p3*nz*p2)),1,p1*p3*nz*p2)));
//               af_print(B);
//               B = moddims(B, p1*p2*p4, p3*nz);
//               B = A * B;
//               B = moddims(B, p1,p2,p4, p3*nz);
//
//         //q=complex(0,q);
//         af_print(B);
//
//
////         array p =randu(1,2,c64);
////         af_print(p);
////         array u = constant(1,2,1,c64);
////         array w = matmul(u,p);
////         af_print(w);
////         array oo = moddims(w,1,4);
////         af_print(oo);
//
//
//
//         //array x = randu(1,2,c64);
//
//
////         af_dtype *ty;
////         af_get_type(ty,mok);
//
//
//
//         //println(summary(mok));
//
////         af_dtype ty;
////         af_get_type(&ty, mok);
//         //z = matmul(constant(1,nsamp,1,c64),z);
//
//
//
//
//        // af_print(mok.isdouble());
//
////        array P = randu(2,2,f64);
////        P = P*[1;1];
////        af_print(P);
////        array B = randu(2,2,2,f32);
////        af_print(B);
////        B = sum(B, 2);
////        af_print(B);
//            std::cout << "dim K(0) = "
//              << B.dims(0)<< std::endl;
//
//            std::cout << "dim K(1) = "
//              << B.dims(1)<< std::endl;
//
//
//              std::cout << "type mok = "
//              << mok.type()<< std::endl;
//        //af_print(A.dims(1));
//
//        //c.T()
////        printf("Element-wise arithmetic\n");
////        array B = sin(A) + 1.5;
////        af_print(B);
////
////        printf("Negate the first three elements of second column\n");
////        B(seq(0, 2), 1) = B(seq(0, 2), 1) * -1;
////        af_print(B);
////
////        printf("Fourier transform the result\n");
////        array C = fft(B);
////        af_print(C);
////
////        printf("Grab last row\n");
////        array c = C.row(end);
////        af_print(c);
////
////        printf("Scan Test\n");
////        dim4 dims(16, 4, 1, 1);
////        array r = constant(2, dims);
////        af_print(r);
////
////        printf("Scan\n");
////        array S = af::scan(r, 0, AF_BINARY_MUL);
////        af_print(S);
////
////        printf("Create 2-by-3 matrix from host data\n");
////        float d[] = { 1, 2, 3, 4, 5, 6 };
////        array D(2, 3, d, afHost);
////        af_print(D);
////
////        printf("Copy last column onto first\n");
////        D.col(0) = D.col(end);
////        af_print(D);
////
////        // Sort A
////        printf("Sort A and print sorted array and corresponding indices\n");
////        array vals, inds;
////        sort(vals, inds, A);
////        af_print(vals);
////        af_print(inds);

    } catch (af::exception& e) {

        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    //return 0;
//}
// d_c - c on the device
//cuComplex * d_cct ;
//// d_c - c on the device
//cudaMalloc (( void **)& d_a , FAr * FAc * sizeof (cuDoubleComplex)); // device
//// d_c - c on the device
//cudaMalloc (( void **)& d_ar , FAr * FAc * sizeof (double)); // device
//// d_c - c on the device
//cudaMalloc (( void **)& d_ai , FAr * FAc * sizeof (double)); // device
//// memory alloc for a
//cudaMalloc (( void **)& d_b , FCr * FCc * sizeof (cuDoubleComplex)); // device
//// memory alloc for a
//cudaMalloc (( void **)& d_br , FCr * FCc * sizeof (double)); // device
//// memory alloc for a
//cudaMalloc (( void **)& d_bi , FCr * FCc * sizeof (double)); // device
//// memory alloc for b
//cudaMalloc (( void **)& d_c , FCc * FCc * sizeof (cuDoubleComplex)); // device
//// memory alloc for t
//cudaMalloc (( void **)& d_ct , FCc * FAc * sizeof (cuDoubleComplex)); // device
//// memory alloc for t
////cudaMalloc (( void **)& d_cct , FAc * FCc * sizeof (cuComplex)); // device
//// memory alloc for c
////cudaMalloc (( void **)& d_cx , FAc * FCc * sizeof (cuComplex)); // device
//// memory alloc for c
//cublasCreate (&handle ); // initialize CUBLAS context
//
//
//
////cudaMemcpy(d_a, b, sizeof(cuComplex)*FAr*FAc, cudaMemcpyHostToDevice);
//cudaMemcpy(d_br, Cr, sizeof(double)*FCr*FCc, cudaMemcpyHostToDevice);
//cudaMemcpy(d_bi, Ci, sizeof(double)*FCr*FCc, cudaMemcpyHostToDevice);
//cudaMemcpy(d_ar, Ar, sizeof(double)*FAr*FAc, cudaMemcpyHostToDevice);
//cudaMemcpy(d_ai, Ai, sizeof(double)*FAr*FAc, cudaMemcpyHostToDevice);
////clock_t time1, time2;
////time1 = clock();
////double	spent_t;
//
////copy data
//	//gpuErrchk(cudaMemcpy(d_imgr , img_inr, n_pixels*sizeof(double), cudaMemcpyHostToDevice));
//	//gpuErrchk(cudaMemcpy(d_imgi , img_ini, n_pixels*sizeof(double), cudaMemcpyHostToDevice));
//    int frow = FCr;
//    int fcol = FCc;
//    int const BLOCK_DIM = 17;
//    dim3 const dimBlock(BLOCK_DIM, BLOCK_DIM);
//    dim3 const dimGrid((frow+BLOCK_DIM-1)/BLOCK_DIM,(fcol+BLOCK_DIM-1)/BLOCK_DIM);
//    matrixcom<<<dimGrid,dimBlock>>>(d_b,d_br,d_bi,frow,fcol);
//
//    frow = FAr;
//    fcol = FAc;
//    int const BLOCK_DIMa = 17;
//    dim3 const dimBlocka(BLOCK_DIMa, BLOCK_DIMa);
//    dim3 const dimGrida((frow+BLOCK_DIMa-1)/BLOCK_DIMa,(fcol+BLOCK_DIMa-1)/BLOCK_DIMa);
//    matrixcom<<<dimGrida,dimBlocka>>>(d_a,d_ar,d_ai,frow,fcol);
//
//
//cublasZgemm(handle,CUBLAS_OP_C,CUBLAS_OP_N,FCc,FCc,FCr,&alpha,d_b,FCr,d_b,FCr,&beta,d_c,FCc);
//cublasDznrm2(handle,FCc*FCc,d_c,1,&result);
//cuDoubleComplex lambda;
//lambda.x = result / FCc * 0.02;
//lambda.y = 0;
//cublasZgemm(handle,CUBLAS_OP_C,CUBLAS_OP_N,FCc,FAc,FCr,&alpha,d_b,FCr,d_a,FAr,&beta,d_ct,FCc);
//
// // Choose a reasonably sized number of threads in each dimension for the block.
//   // int const threadsPerBlockEachDim = 16;
// int const BLOCK_DIMc = 17;
// dim3 const dimBlockc(BLOCK_DIMc, BLOCK_DIMc);
// //dim3 const dimGrid((int)ceil(FAc/dimBlock.x),(int)ceil(FCc/dimBlock.y));
// dim3 const dimGridc((FCc+BLOCK_DIMc-1)/BLOCK_DIMc,(FCc+BLOCK_DIMc-1)/BLOCK_DIMc);
// matrixAdd<<<dimGridc,dimBlockc>>>(d_c,lambda,FCc);
//
// //////////////////////invers matrix////////////////////////////////////
////    cuComplex** adL;
////    cuComplex** adC;
////    //cuComplex* dL;
////    cuComplex* dC;
////    int* dLUPivots;
////    int* dLUInfo;
////
////    size_t szA = FCc * FCc * sizeof(cuComplex);
////
////    CUDA_CALL(cudaMalloc(&adL, sizeof(cuComplex*)), "Failed to allocate adL!");
////    CUDA_CALL(cudaMalloc(&adC, sizeof(cuComplex*)), "Failed to allocate adC!");
////    //CUDA_CALL(cudaMalloc(&dL, szA), "Failed to allocate dL!");
////    CUDA_CALL(cudaMalloc(&dC, szA), "Failed to allocate dC!");
////    CUDA_CALL(cudaMalloc(&dLUPivots, FCc * sizeof(int)), "Failed to allocate dLUPivots!");
////    CUDA_CALL(cudaMalloc(&dLUInfo, sizeof(int)), "Failed to allocate dLUInfo!");
////
////    //CUDA_CALL(cudaMemcpy(dL, L, szA, cudaMemcpyHostToDevice), "Failed to copy to dL!");
////    CUDA_CALL(cudaMemcpy(adL, &d_c, sizeof(cuComplex*), cudaMemcpyHostToDevice), "Failed to copy to adL!");
////    CUDA_CALL(cudaMemcpy(adC, &dC, sizeof(cuComplex*), cudaMemcpyHostToDevice), "Failed to copy to adC!");
////
////    CUBLAS_CALL(cublasCgetrfBatched(handle, FCc, adL, FCc, dLUPivots, dLUInfo, 1), "Failed to perform LU decomp operation!");
////    CUDA_CALL(cudaDeviceSynchronize(), "Failed to synchronize after kernel call!");
////
////    CUBLAS_CALL(cublasCgetriBatched(handle, FCc, (const cuComplex **)adL, FCc, dLUPivots, adC, FCc, dLUInfo, 1), "Failed to perform Inverse operation!");
////    CUDA_CALL(cudaDeviceSynchronize(), "Failed to synchronize after kernel call!");
// //////////////////////akhir invers matrix//////////////////////////////
////cublasZgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,FCc,FAc,FCc,&alpha,dC,FCc,d_ct,FAc,&beta,d_cct,FCc);
////lambda = norm(AtA, 'fro') / size(src_mtx,2) * 0.02; % Regularization parameter, from Miki Lustig's code
////ker{ker_idx}.ker = (AtA + eye(size(AtA))*lambda) \ (src_mtx' * tgt_mtx); % Dim: [kersz(1)*kersz(2)*nc, nc*nz]
//
//
////time2 = clock();
////spent_t = (double) (time2-time1) / CLOCKS_PER_SEC;
////printf("The time spent for multiply : %1.8f  second \n",spent_t);
//////printf("AtA ke 0 : %6.8f \n",d_c);
////
////printf("ini lambda : %6.8f \n",lambda.x);
//
//
////cudaMemcpy(cc, d_c, sizeof(cuDoubleComplex)*FCc*FCc, cudaMemcpyDeviceToHost);
////cudaMemcpy(cct, d_ct, sizeof(cuDoubleComplex)*FCc*FAc, cudaMemcpyDeviceToHost);
//
//
//    //convec(FAr,FAc,FCr, FCc);
//
//
//
//   FCc =6;
//   int rows=FCc;
//   int cols=FCc;
//

//
//  ////////////////////////////////////////////////////
//  int rowsc=FCc;
//
//   int colsc=FAc;
//
//   plhs[1] = mxCreateDoubleMatrix(rowsc, colsc, mxCOMPLEX);
//  double *bufferctr = (double *)mxGetPr(plhs[1]);
//  double *buffercti = (double *)mxGetPi(plhs[1]);
////////////////////////////////////////////////////////////////
//
//   //Mat data is float, and mxArray uses double, so we need to convert.
//  //plhs[0]=mxCreateDoubleMatrix(cols, rows, mxREAL);
//
////plhs[0] = mxCreateDoubleMatrix(1, len, mxREAL);
////memcpy(mxGetPr(plhs[0]), cc.x, cols*rows*sizeof(cuComplex));
//memcpy(mxGetPi(plhs[0]), cc.y, cols*rows*sizeof(cuComplex));
//   double *buffercr=(double*)mxGetPr(plhs[0]);
//   double *bufferci=(double*)mxGetPi(plhs[0]);
//   for(int i=0; i<rows; i++){
//       for(int j=0; j<cols; j++){
//           buffercr[i*(cols)+j]= (cc[i*(cols)+j].x);
//           bufferci[i*(cols)+j]= (cc[i*(cols)+j].y);
//           //printf("hasil cc : %1.8f + i %1.8f \n",(cc[i*(cols)+j].x),(cc[i*(cols)+j].y));
//
//       }
//   }

  // pisah(d_c,rows,cols,buffercr,bufferci);


//   for(int i=0; i<rowsc; i++){
//       for(int j=0; j<colsc; j++){
//           bufferctr[i*(colsc)+j]= (cct[i*(colsc)+j].x);
//           buffercti[i*(colsc)+j]= (cct[i*(colsc)+j].y);
//           //printf("hasil cc : %1.8f + i %1.8f \n",(cc[i*(cols)+j].x),(cc[i*(cols)+j].y));
//
//       }
//   }

   // pisah(d_ct,rowsc,colsc,bufferctr,buffercti);


//cudaFree ( d_a );
//cudaFree ( d_ar );
//cudaFree ( d_ai );
//// free device memory
//cudaFree ( d_b );
//cudaFree ( d_br );
//cudaFree ( d_bi );
//// free device memory
////cudaFree ( d_c );
//// free device memory
////cudaFree ( d_ct );
//
//
//    //CUDA_CALL(cudaFree(adL), "Failed to free adL!");
//    //CUDA_CALL(cudaFree(adC), "Failed to free adC!");
//    //CUDA_CALL(cudaFree(dL), "Failed to free dL!");
//    //CUDA_CALL(cudaFree(dC), "Failed to free dC!");
//    //CUDA_CALL(cudaFree(dLUPivots), "Failed to free dLUPivots!");
//    //CUDA_CALL(cudaFree(dLUInfo), "Failed to free dLUInfo!");
//// free device memory
//cublasDestroy ( handle );
////cudaDeviceReset();
////cudaDeviceSynchronize();
//// destroy CUBLAS context
//free( a );
// free host memory
//free( b );
// free host memory
//free( cc );


//printf("%f  %f \n", a[0].x,a[3].x);

//printf("sayayayayay1");
//    A = mxGPUCreateFromMxArray(prhs[0]);
//    C = mxGPUCreateFromMxArray(prhs[1]);
//    //m = mxGPUCreateFromMxArray(prhs[2]);
//    //n = mxGPUCreateFromMxArray(prhs[3]);
//    //pp = mxGPUCreateFromMxArray(prhs[4]);
//    // mxGPUArray const * A = mxGPUCreateFromMxArray(prhs[0]);
//   // printf("sayayayayay4");
//
//    /*
//     * Verify that A really is a cuDoubleComplex array before extracting the pointer.
//     */
//    if (mxGPUGetClassID(A) != mxDOUBLE_CLASS) {
//        mexErrMsgIdAndTxt(errId, errMsg);
//    }
//
//    /*
//     * Now that we have verified the data type, extract a pointer to the input
//     * data on the device.
//     */
//    d_A = (cuDoubleComplex const *)(mxGPUGetDataReadOnly(A));
//    d_C = (cuDoubleComplex const *)(mxGPUGetDataReadOnly(C));
//    //d_m = (int const *)(mxGPUGetDataReadOnly(m));
//    //d_n = (int const *)(mxGPUGetDataReadOnly(n));
//    //d_pp = (int const *)(mxGPUGetDataReadOnly(pp));
////    d_m = (int)*mxGetPr(prhs[2]);
////    d_n = (int)*mxGetPr(prhs[3]);
////    d_pp = (int)*mxGetPr(prhs[4]);
//

   // d_pp = dim_arrayC[0];
//printf("sayayayayay2");
 //printf("FAr : %2.0d  adalah \n",FAr);
  //printf("FAc : %2.0f  adalah \n",FAc);

//    /* Create a GPUArray to hold the result and get its underlying pointer. */
//    mwSize dims[] = {dim_arrayA[0],dim_arrayC[1]};
//    //mwSize dims[] = {2,2};
//    //printf("maaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaauuuuuuuuuuuuuuuuu");
//   // printf("m x pp : %2d  adalah \n",55555);
//    B = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(A),
//                            dims,
//                            mxGPUGetClassID(A),
//                            mxGPUGetComplexity(A),
//                            MX_GPU_DO_NOT_INITIALIZE);
//    d_B = (cuDoubleComplex *)(mxGPUGetData(B));

    /*
     * Call the kernel using the CUDA runtime API. We are using a 1-d grid here,
     * and it would be possible for the number of elements to be too large for
     * the grid. For this example we are not guarding against this possibility.
     */
    //N = (int)(mxGPUGetNumberOfElements(A));
    //blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    //TimesTwo<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);
    //printf("Ini Class dari d_A \n");

    //class(d_A);

    /* Wrap the result up as a MATLAB gpuArray for return. */
    //plhs[0] = mxGPUCreateMxArrayOnGPU(B);
    //plhs[0] = 0;

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
//    mxGPUDestroyGPUArray(A);
//    mxGPUDestroyGPUArray(C);
//    //mxGPUDestroyGPUArray(m);
//    //mxGPUDestroyGPUArray(n);
//    //mxGPUDestroyGPUArray(pp);
//    mxGPUDestroyGPUArray(B);
}
//
//void __global__ matrixdo(cuDoubleComplex * const c,double * const re,double * const im, int const fr, int const fc) {
// int col = blockIdx.x * blockDim.x + threadIdx.x;
// int row = blockIdx.y * blockDim.y + threadIdx.y;
// //int index = row + col * N;
// int index = col + row * fr;
// if (col < fr && row < fc) {
// //cx[index] = c[index];
//
// //if(index<(fr*fc)){
//
//  //re[index] = index ;
//  //im[index] = index ;
//  re[index] = c[index].x ;
//  im[index] = c[index].y ;
//  //}
////
//  //c[index].x = (double)index;
//  //c[index].y = 0;
//  //printf("ini c[index].x : %6.8f \n",c[index].x);
//
// }
//}
//
//
//void pisah(cuDoubleComplex * d_col, int cr, int cc, double * buffercr, double * bufferci)
//{
//    int col_size = cr;
//    int total_size = cc;
//    //cuDoubleComplex * d_col = c;
//    double * datar;
//    double * datai;
//
//
//    gpuErrchk(cudaMalloc(&datar, col_size*total_size*sizeof(double)));
//    gpuErrchk(cudaMalloc(&datai, col_size*total_size*sizeof(double)));
//
//
//    int frowdo = col_size;
//    int fcoldo = total_size;
//    int const BLOCK_DIMd = 17;
//    dim3 const dimBlockd(BLOCK_DIMd, BLOCK_DIMd);
//    dim3 const dimGridd((frowdo+BLOCK_DIMd-1)/BLOCK_DIMd,(fcoldo+BLOCK_DIMd-1)/BLOCK_DIMd);
//    //printf("ini frowdo : %6d ini coldo : %6d \n", frowdo,fcoldo);
//    matrixdo<<<dimGridd,dimBlockd>>>(d_col,datar,datai,frowdo,fcoldo);
//    gpuErrchk(cudaMemcpy(buffercr, datar, col_size*total_size*sizeof(double), cudaMemcpyDeviceToHost));
//    gpuErrchk(cudaMemcpy(bufferci, datai, col_size*total_size*sizeof(double), cudaMemcpyDeviceToHost));
//
//
//
//	gpuErrchk(cudaFree(d_col));
//	gpuErrchk(cudaFree(datar));
//	gpuErrchk(cudaFree(datai));
//
//
//
//}
