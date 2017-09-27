#include <stdio.h>
#include <time.h>
#define n 1024
#define x_threads 8
#define y_threads 8

__global__ void gpu_matMult(int *a, int *b, int *c, int n){
	int k, sum = 0;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	if (i < n && j < n){
    for (k = 0; k < n; k++) {
      sum += a[j * n + k] * b[k * n + i];
    }
    c[j * n + i] = sum;
  }
}

int main(int argc, char const *argv[])
{

	size_t bytes = n * n * sizeof(int);

	double timeGPU;
  int *h_a, *h_b, *h_c, *d_a, *d_b, *d_c;
  h_a = (int *)malloc(bytes);
  h_b = (int *)malloc(bytes);
  h_c = (int *)malloc(bytes);

  cudaMalloc((void **) &d_a, bytes);
 	cudaMalloc((void **) &d_b, bytes);
 	cudaMalloc((void **) &d_c, bytes);

 	for (int i = 0; i < n; i++) {
  	int cont = 0;
  	for (int j = 0; j < n; j++) {
   		h_a[i][j] = cont * n;
   		h_b[i][j] = cont * n;
   		cont++;
  	}
  }

  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
 	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

 	dim3 thredsPerBlock(x_threads, y_threads);
 	dim3 numBlocks((int)ceil((float)n/x_threads), (int)ceil((float)n/y_threads));

 	clock_t startGPU  = clock();
 	gpu_matMult<<<numBlocks, thredsPerBlock>>>(d_a, d_b, d_c, n);

 	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
 	timeGPU = ((double)(clock() - startGPU))/CLOCKS_PER_SEC;

 	printf("tiempo GPU = %f s\n",timeGPU);

	return 0;
}
