#include <stdio.h>
#include <time.h>
#define N 512

void Matriz_CPU_Mult(int *h_a, int *h_b, int *h_c) {
	int n,m;
	for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
   		int sum = 0;
      for (int k = 0; k < N; k++) {
        m = h_a[i][k];
        n = h_b[k][j];
        sum += m * n;
      }
   	h_c[i][j] = sum;
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
  double timeGPU; //, timeCPU;
	int **h_a, **h_b, **h_c;
 	int *d_a, *d_b, *d_c;
 	int cont,i,j;

  int size = N * sizeof(int);

  h_a = (int**)malloc(size);
  h_b = (int**)malloc(size);
  h_c = (int**)malloc(size);

  cudaMalloc((void **) &d_a, size);
  cudaMalloc((void **) &d_b, size);
  cudaMalloc((void **) &d_c, size);

  //inicializacion
	for (i = 0; i < N; i++) {
  	cont = 0;
  	for (j = 0; j < N; j++) {
   		h_a[i][j] = cont;
   		h_b[i][j] = cont;
   		cont++;
  	}
  }

  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
 	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  //int threadsPerBlock(16);
  //int numBlocks(N/threadsPerBlock);
  dim3 threadsPerBlock(32, 32);
 	dim3 numBlocks(N/threadsPerBlock.x, N/threadsPerBlock.y);
  
	clock_t startGPU  = clock();
  Matriz_GPU_Mult<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c);
	timeGPU = ((double)(clock() - startGPU))/CLOCKS_PER_SEC;
  
  cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
	
  /*
  clock_t startCPU = clock();
  Matriz_CPU_Mult(A, B, C);
	timeCPU = ((double)(clock() - startCPU))/CLOCKS_PER_SEC;
  */

  free(h_a);
  free(h_b);
  free(h_c);

  cudaFree(d_a);
 	cudaFree(d_b);
 	cudaFree(d_c);

  // tiempos de ejecucion
  printf("tiempo GPU = %f s\n",timeGPU);
	//printf("\ntiempo CPU = %f s\n",timeCPU);
  
  return 0;
}
