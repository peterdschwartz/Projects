
__global__
void sposv_batched_kernel(int n, int batch, float *dA);

////////////////////////////////////////////////////////////////////////////////
extern "C"
void sposv_batched(int n, int batch, float *dA, cudaStream_t stream)
{
  int divb = 4; // divides up the batch number
  // int divmat = 1; // divides up individual nxn matrix
  dim3 dimBlock(batch/divb, 1, 1);
  // dim3 dimBlock(1, 1, 1);
  dim3 dimGrid(divb, 1, 1);
  sposv_batched_kernel<<<dimGrid, dimBlock, 0, stream>>>(n, batch, dA);
}

////////////////////////////////////////////////////////////////////////////////
__global__
void sposv_batched_kernel(int N, int batch, float *dA)
{

  int m;
  int n;
  int k;// = blockIdx.y*blockDim.y+threadIdx.y;
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  // Batched Cholesky factorization.
  //for (i = 0; i < batch; i++) {
  if(i<batch){
    float *pA = &dA[i*N*N];
    
    // Single Cholesky factorization.
    for (k = 0; k < N; k++) {
    //if(k<N){
      // Panel factorization.
      pA[k*N+k] = sqrtf(pA[k*N+k]);
      for (m = k+1; m < N; m++)
	pA[k*N+m] /= pA[k*N+k];
      
      // Update of the trailing submatrix.
      for (n = k+1; n < N; n++)
	for (m = n; m < N; m++)
	  pA[n*N+m] -= (pA[k*N+n]*pA[k*N+m]);
    }
  }
}
