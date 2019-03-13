#include <stdio.h>
//#include <cutil.h>
#include "cublas.h"
//#include </opt/eecs/Matlab/R2015b/toolbox/rtw/rtwdemos/crl_demo/cblas.h>

//=============================================================================

extern "C" int sgemm_(char *, char *, int *, int *, int *, float *, float *, 
                      int *, float *, int *, float *, float *, int *);
extern "C" int sgesvd_(char *, char *, int *, int *, float *, int *, float *, 
                       float *, int *, float *, int *, float *, int *, int *);
extern "C" void sgeqrf_(int*, int*, float*, int*, float*, float*, int*, int*);
extern "C" int scopy_(int *, float*, int *, float*, int *);
extern "C" int strmm_(char*, char *, char*, char *, int *, int *, float *, 
                      float *, int *, float *, int *);
extern "C" int strsm_(char *, char *, char *, char *, int *, int *, 
                      float *, float *, int *, float *, int *);
extern "C" int ssyrk_(char *, char *, int *, int *, float *, float *, 
                      int *, float *, float *, int *);
//=============================================================================

void chol_qr_it(int m, int n, float *A, int lda, float *R){
    int i = 0, k, j, info, lwork = n*n, n2 = n*n, one = 1;
    float *G, *U, *S, *VT, *vt, *tau, *work;  
    float cn = 200.f, alpha = 1.f, zero = 0.f, mins, maxs;
    
    G    = (float*)malloc(n * n * 4);
    VT   = (float*)malloc(n * n * 4);
    S    = (float*)malloc(    n * 4);
    work = (float*)malloc(lwork * 4);
    tau  = (float*)malloc(    n * 4);  

    do {
      i++;

      sgemm_("t", "n", &n, &n, &m, &alpha, A, &m, A, &m, &zero, G, &n);
      //ssyrk_('l', 't', &n, &m, &alpha, A, &m, &zero, G, &n);
      //for(j=0; j<n; j++)
      //  for(k=0; k<j; k++)
      //     G[j*n+k] = G[k*n+j];

      sgesvd_("n", "a", &n, &n, G, &n, S, U, &n, VT, &n, work, &lwork, &info);

      mins = 100.f, maxs = 0.f;
      for(k=0; k<n; k++){
        S[k] = sqrt(S[k]);

	if (S[k] < mins)  mins = S[k];
	if (S[k] > maxs)  maxs = S[k];
      }

      for(k=0; k<n;k++){
        vt = VT + k*n;
        for(j=0; j<n; j++)
          vt[j]*=S[j];
      } 
      sgeqrf_(&n, &n, VT, &n, tau, work, &lwork, &info);

      if (i==1)
        scopy_(&n2, VT, &one, R, &one);
      else
        strmm_("l", "u", "n", "n", &n, &n, &alpha, VT, &n, R, &n);
      
      strsm_("r", "u", "n", "n", &m, &n, &alpha, VT, &n, A, &m);    

      if (mins > 0.00001f) 
        cn = maxs/mins;

      fprintf(stderr, "\nIteration %d, cond num = %f just cpu \n", i, cn);
    } while (cn > 100.f);

    free(G);
    free(VT);
    free(S);
    free(work);
    free(tau);

}

//=============================================================================

void chol_qr_it_GPU(int m, int n, float *d_A, int lda, float *G, float *R, 
                 float *work, int lwork){
  int i = 0, k, j, info, n2 = n*n, one = 1,lworksvd=n*n;
    float *U, *S, *VT, *vt, *tau,*worksvd;
    float cn = 200.f, alpha = 1.f, zero = 0.f, mins, maxs;
    // cublasHandle_t handle;
    //cublasCreate(&handle);

    VT   = (float*)malloc(n * n * 4);
    S    = (float*)malloc(    n * 4);
    tau  = (float*)malloc(    n * 4);
    worksvd = (float*)malloc(lworksvd* 4);
    do {
        i++;
	
      // cublasSgemm(&handle,CUBLAS_OP_T,CUBLAS_OP_N, n, n, m, &alpha, *d_A, m, *d_A, m, &zero, *G, n);
      cublasSgemm('T','N', n, n, m, alpha, d_A, m, d_A, m, zero, G, n);

      cublasGetVector(n*n,sizeof(float),G,1,work,1 );

      sgesvd_("n", "a", &n, &n, work, &n, S, U, &n, VT, &n, worksvd, &lworksvd, &info);
                  fprintf(stderr,"hi \n");    
      // sgesvd_( ... );

      mins = 100.f, maxs = 0.f;
      for(k=0; k<n; k++){
        S[k] = sqrt(S[k]);

        if (S[k] < mins)  mins = S[k];
        if (S[k] > maxs)  maxs = S[k];
      }

      for(k=0; k<n;k++){
        vt = VT + k*n;
        for(j=0; j<n; j++)
          vt[j]*=S[j];
      }
      sgeqrf_(&n, &n, VT, &n, tau, work, &lwork, &info);
      //      sgeqrf_( ... );

      if (i==1)
        scopy_(&n2, VT, &one, R, &one);	
      // scopy_( ... );
      else
	strmm_("l", "u", "n", "n", &n, &n, &alpha, VT, &n, R, &n);
      
      cublasSetVector(n*n,sizeof(float),R,1,G,1 );
      cublasStrsm('r', 'u', 'N', 'N', m, n, alpha, G, n, d_A, m);    

      if (mins > 0.00001f)
        cn = maxs/mins;
      
      fprintf(stderr, "\nIteration %d, cond num = %f \n", i, cn);
       } while (cn > 100.f);
    //    cublasDestroy(&handle);
    free(VT);
    free(S);
    free(tau);
    free(worksvd);
    }

//=============================================================================
