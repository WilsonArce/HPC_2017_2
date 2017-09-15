#include <stdio.h>
#include <time.h>
#define N 1024

void Matriz_CPU_Mult(int A[N][N], int B[N][N], int C[N][N]) {
	int n,m;
	for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
   		int sum = 0;
      for (int k = 0; k < N; k++) {
        m = A[i][k];
        n = B[k][j];
        sum += m * n;
      }
   	C[i][j] = sum;
  	}
 	}
}

__global__ void Matriz_GPU_Mult(int *a, int *b, int *c) {
	int k, sum = 0;
	int i = blockIdx.x * blockDim.x + threadIdx.x; 
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < N && j < N) {
    for (k = 0; k < N; k++) {
      sum += a[j * N + k] * b[k * N + i];
    }
    c[j * N + i] = sum;
  }
}

int main() {
  double timeGPU, timeCPU;
	int A[N][N], B[N][N], C[N][N];
 	int *d_a, *d_b, *d_c;
 	int cont,i,j;

  //inicializacion
	for (i = 0; i < N; i++) {
  	cont = 0;
  	for (j = 0; j < N; j++) {
   		A[i][j] = cont;
   		B[i][j] = cont;
   		cont++;
  	}
  }

  int size = N * N * sizeof(int);

	cudaMalloc((void **) &d_a, size);
 	cudaMalloc((void **) &d_b, size);
 	cudaMalloc((void **) &d_c, size);

  cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice);
 	cudaMemcpy(d_b, B, size, cudaMemcpyHostToDevice);

  //int threadsPerBlock(16);
  //int numBlocks(N/threadsPerBlock);
  dim3 threadsPerBlock(16, 16);
 	dim3 numBlocks(N/threadsPerBlock.x, N/threadsPerBlock.y);
  
	clock_t startGPU  = clock();
  Matriz_GPU_Mult<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c);
	timeGPU = ((double)(clock() - startGPU))/CLOCKS_PER_SEC;
  
  cudaMemcpy(C, d_c, size, cudaMemcpyDeviceToHost);
	
  clock_t startCPU = clock();
  Matriz_CPU_Mult(A, B, C);
	timeCPU = ((double)(clock() - startCPU))/CLOCKS_PER_SEC;
  
  cudaFree(d_a);
 	cudaFree(d_b);
 	cudaFree(d_c);

  // tiempos de ejecucion
  printf("tiempo GPU = %f s",timeGPU);
	printf("\ntiempo CPU = %f s",timeCPU);
  
  return 0;
}