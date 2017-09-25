#include <stdio.h>
#define MWIDTH 4096
#define MTILE 16
#define BWIDTH 16

__global__ void gpu_matrixMul(int *a, int *b, int *c, int Width, int tile_width){

  int start_row = (blockDim.y*blockIdx.y + threadIdx.y)*tile_width;
  int end_row = start_row + tile_width;
  int start_col = (blockDim.x*blockIdx.x + threadIdx.x)*tile_width;
  int end_col = start_col + tile_width;

  for (int row = start_row; row < end_row; row++) {
    for(int col = start_col; col < end_col; col++) {
      float sum = 0;
      for (int k = 0; k < Width; k++) {
        sum += a[row * Width + k]*b[k * Width + col];
      }
      c[row*Width+col] = sum;
    }
  }
}



int main(){

  int *h_a, *h_b, *h_c, *d_a, *d_b, *d_c;
  h_a = (int *)malloc(MWIDTH*MWIDTH*sizeof(int));
  h_b = (int *)malloc(MWIDTH*MWIDTH*sizeof(int));
  h_c = (int *)malloc(MWIDTH*MWIDTH*sizeof(int));
  cudaMalloc(&d_a, MWIDTH*MWIDTH*sizeof(int));
  cudaMalloc(&d_b, MWIDTH*MWIDTH*sizeof(int));
  cudaMalloc(&d_c, MWIDTH*MWIDTH*sizeof(int));

  for (int i = 0; i < MWIDTH*MWIDTH; i++) {
    h_a[i] = 1;
    h_b[i] = 1;
    h_c[i] = 0;}

  cudaMemcpy(d_a, h_a, MWIDTH*MWIDTH*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, MWIDTH*MWIDTH*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(d_c, 0, MWIDTH*MWIDTH*sizeof(int));

  gpu_matrixMul<<<dim3((MWIDTH/(MTILE*BWIDTH)), (MWIDTH/(MTILE*BWIDTH))), dim3(BWIDTH,BWIDTH)>>>(d_a, d_b, d_c, MWIDTH, MTILE);
  cudaMemcpy(h_c, d_c, MWIDTH*MWIDTH*sizeof(int), cudaMemcpyDeviceToHost);
  for (int i=0; i < MWIDTH*MWIDTH; i++)
    if (h_c[i] != MWIDTH) {printf("Mismatch at offset %d, was: %d, should be: %d\n", i, h_c[i], MWIDTH); return 1;}
  printf("Success!\n");
  return 0;
}