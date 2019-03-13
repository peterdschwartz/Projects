// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
//#include </opt/eecs/Matlab/R2015b/toolbox/rtw/rtwdemos/crl_demo/cblas.h>
// includes, project


#include "cublas.h"
//=============================================================================
extern "C" void  saxpy_(const int *, const float *, const float *, const int *, 
                             const float *, const int *);
extern "C" float snrm2_(const int *, const float *, const int *);
extern "C" float isamax_(const int *, const float *, const int *);
extern "C" void      sgeqrf_(int*,int*,float*,int*,float*,float*,int*,int*);
extern "C" int strmm_(char*, char *, char*, char *, int *, int *, float *,
                      float *, int *, float *, int *);
extern "C" int sgemm_(char *, char *, int *, int *, int *, float *, float *,
                      int *, float *, int *, float *, float *, int *);

void chol_qr_it(int m, int n, float *A, int lda, float *R);
void chol_qr_it_GPU(int m, int n, float *d_A, int lda, float *d_G, float *R, 
                 float *h_work, int lwork);
//=============================================================================

///////////////////////////////////////////////////////////////////////////////
// Program main
///////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
  //CUT_DEVICE_INIT();

    unsigned int timer = 0;

    /* Matrix size */
    int N, M;                 // NxM would be the size of the matrices 
                              // (M columns) that we would orthogonalize
    float *d_A, *d_G;         // d_A is array for A on the device (GPU)
    float *h_work, *h_tau;    // work space and array tau on the host
    float *h_A, *h_Q1, *h_Q2; // These would be the same NxM matrices 
    float *h_R, *h_G;

    int info[1], lwork, i;

    N  = 131072;
    M  = 128;    

    if (argc != 1)
    for(i = 1; i<argc; i++){	
      if (strcmp("-N", argv[i])==0)
         N = atoi(argv[++i]);
      else if (strcmp("-M", argv[i])==0)
         M = atoi(argv[++i]);
    }
    printf("\nUsage: \n");
    printf("  chol_qr_it -N %d -M %d\n\n", N, M);

    lwork = 2*N*M;

    int n2 = N * M;

    /* Initialize CUBLAS */
    cublasInit();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    /* Allocate host memory for the matrix */
    h_A  = (float*)malloc(n2 * sizeof( h_A[0]));
    h_Q1 = (float*)malloc(n2 * sizeof(h_Q1[0]));
    h_Q2 = (float*)malloc(n2 * sizeof(h_Q2[0]));
   
    h_G = (float*)malloc(M*M * sizeof(h_G[0]));
    h_R = (float*)malloc(M*M * sizeof(h_R[0]));
  
    cudaMallocHost( (void**)&h_work, lwork*4);
    cudaMallocHost( (void**)&h_work, lwork*4);
    h_tau = (float*)malloc(N * sizeof(h_tau[0]));
   
    /* Take a random matrix h_A = h_Q1 = h_Q2 */
    for (i = 0; i < n2; i++) {
        h_A[i] = h_Q1[i] = h_Q2[i] = rand() / (float)RAND_MAX;
    }

    /* Allocate device memory for the matrices */
    cublasAlloc(n2, sizeof(d_A[0]), (void**)&d_A);
    cublasAlloc(M*M, sizeof(d_G[0]), (void**)&d_G);

    // create and start timer
    /* timer = 0; */
    // /* CUT_SAFE_CALL(cutCreateTimer(&timer)); */
    // /* CUT_SAFE_CALL(cutStartTimer(timer)); */

    /* =====================================================================
         Performs QR on CPU using LAPACK 
       ===================================================================== */
    sgeqrf_(&N, &M, h_A, &N, h_tau, h_work, &lwork, info);
    if (info[0] < 0)  
       printf("Argument %d of sgeqrf had an illegal value.\n", -info[0]);     

    // stop and destroy timer
    // CUT_SAFE_CALL(cutStopTimer(timer));
    // printf("CPU Processing time: %f (ms) \n", cutGetTimerValue(timer));
    //printf("Speed: %f GFlops \n", 4.*N*M*M/
    //      (3.*1000000*cutGetTimerValue(timer)));
    //CUT_SAFE_CALL(cutDeleteTimer(timer));


    /* Initialize the device matrix with the host matrices */
    cublasSetVector(n2, sizeof(h_Q2[0]), h_Q2, 1, d_A, 1);

    //timer = 0;
    //    CUT_SAFE_CALL(cutCreateTimer(&timer));
    //CUT_SAFE_CALL(cutStartTimer(timer));

    /* =====================================================================
         Performs orthogonalization on CPU using chol_qr_it
       ===================================================================== */
    cudaEventRecord(start);
    chol_qr_it(N, M, h_Q2, N, h_R);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // stop and destroy timer
    // CUT_SAFE_CALL(cutStopTimer(timer));
     printf("\n\nCPU Processing time: %f (ms) \n", milliseconds);
     printf("Speed: %f GFlops \n", 4.*N*M*M/(3.*1000000*milliseconds)); 
    // CUT_SAFE_CALL(cutDeleteTimer(timer));
    
    float one = 1.f, zero = 0.f;
    const int MM=M*M,nn=n2;
    const int myint=1;
    const float minusone = -1.0f;
    
    sgemm_("t", "n", &M, &M, &N, &one, h_Q2, &N, h_Q2, &N, &zero, h_G, &M);
    fprintf(stderr, "\nIteration just cpu \n");
    for(i=0; i<M*M; i+=(M+1)) h_G[i] -= one;
    printf(" ||I - Q'Q||_F = %e, ||I-Q'Q||_max = %e \n", snrm2_(&MM, h_G, &myint), isamax_(&MM, h_G, &myint));
    fprintf(stderr, "\nIteration just cpu \n");
    strmm_("r", "u", "n", "n", &N, &M, &one, h_R, &M, h_Q2, &N);
    saxpy_(&n2, &minusone, h_Q1, &myint, h_Q2, &myint);
    printf(" ||A - Q R||_F = %e \n",snrm2_(&nn, h_Q2, &myint));    

    // chol_qr on GPU
    timer = 0;
    /* CUT_SAFE_CALL(cutCreateTimer(&timer)); */
    /* CUT_SAFE_CALL(cutStartTimer(timer)); */

    /* =====================================================================
         Performs orthogonalization on CPU-GPU using chol_qr_it
       ===================================================================== */
    //fprintf(stderr,"Madeit to gpuasdlkfljk \n");
    cudaEventRecord(start);
    chol_qr_it_GPU(N, M, d_A, N, d_G, h_R, h_work, lwork);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    // stop and destroy timer
    /* CUT_SAFE_CALL(cutStopTimer(timer)); */
    printf("\n\nGPU Processing time: %f (ms) \n", milliseconds);
     printf("Speed: %f GFlops \n", 4.*N*M*M/(3.*1000000*milliseconds));
    /* CUT_SAFE_CALL(cutDeleteTimer(timer)); */

    /* Read the result back */
    cublasGetVector(n2, sizeof(h_Q2[0]), d_A, 1, h_Q2, 1);

    sgemm_("t", "n", &M, &M, &N, &one, h_Q2, &N, h_Q2, &N, &zero, h_G, &M);
    for(i=0; i<M*M; i+=(M+1)) h_G[i] -= one;
    printf(" ||I - Q'Q||_F = %e, ||I-Q'Q||_max = %e \n",
	   snrm2_(&MM, h_G, &myint), isamax_(&MM, h_G, &myint));

    strmm_("r", "u", "n", "n", &N, &M, &one, h_R, &M, h_Q2, &N);
    saxpy_(&n2, &minusone, h_Q1, &myint, h_Q2, &myint);
    printf(" ||A - Q R||_F = %e \n",
            snrm2_(&nn, h_Q2, &myint));

    /* Memory clean up */
    free(h_A);
    free(h_Q1);
    free(h_Q2);
    free(h_R);
    free(h_G);
    // CUDA_SAFE_CALL( cublasFree(h_work) );
    cublasFree(h_work);
    free(h_tau);

    cublasFree(d_G);
    cublasFree(d_A);

    /* Shutdown */
    cublasShutdown();

    //CUT_EXIT(argc, argv);
}
