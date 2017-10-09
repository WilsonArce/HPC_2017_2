#include <stdio.h>
#include <time.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define chSize 3

#define N 10

__global__ void Matriz_GPU_Mult(double *a, double *b, double *c) {
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

/*
__global__ void gpuGrayScale(int *A, float *B, int cols, int rows){
  int tidx = (blockDim.x * blockIdx.x + threadIdx.x) + chSize;
  int tidy = blockDim.y * blockIdx.y + threadIdx.y;

  float r,g,b;

  printf("%d,%d", tidx, tidy);

  for(int row = tidy; row < rows; row++){
    for(int col = tidx; col < cols; col += chSize){
      r = A[row * cols + col];
      g = A[row * cols + col + 1];
      b = A[row * cols + col + 2];
      
      for(int k = chSize - 1; k >= 0; k--){
        B[row * cols + col - k] = (r * 0.299 + g * 0.587 + b * 0.114);
      }
    }
  }

}
*/
int main(int argc, char** argv )
{
  double timeGPU; //, timeCPU;
  double A[N][N], B[N][N], C[N][N];
  double *d_a, *d_b, *d_c;
  int cont,i,j;

  //inicializacion
  for (i = 0; i < N; i++) {
    cont = 0;
    for (j = 0; j < N; j++) {
      A[i][j] = cont * 987;
      B[i][j] = cont * 987;
      cont++;
    }
  }

  size_t bytes = N * sizeof(double);

  cudaMalloc((void **) &d_a, bytes);
  cudaMalloc((void **) &d_b, bytes);
  cudaMalloc((void **) &d_c, bytes);

  cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, bytes, cudaMemcpyHostToDevice);

  //int threadsPerBlock(16);
  //int numBlocks(N/threadsPerBlock);
  dim3 threadsPerBlock(3, 3);
  dim3 numBlocks((int)ceil((float)N/threadsPerBlock.x), (int)ceil((float)N/threadsPerBlock.y));
  
  clock_t startGPU  = clock();
  Matriz_GPU_Mult<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c);
  timeGPU = ((double)(clock() - startGPU))/CLOCKS_PER_SEC;
  
  cudaMemcpy(C, d_c, bytes, cudaMemcpyDeviceToHost);
  
  /*
  clock_t startCPU = clock();
  Matriz_CPU_Mult(A, B, C);
  timeCPU = ((double)(clock() - startCPU))/CLOCKS_PER_SEC;
  */

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  // tiempos de ejecucion
  printf("tiempo GPU = %f s\n",timeGPU);
  cout << C[0] << endl;

/*
  if ( argc != 2 )
  {
    printf("usage: DisplayImage.out <Image_Path>\n");
    return -1;
  }

  Mat image;
  image = imread( argv[1], 1 );

  if ( !image.data )
  {
    printf("No image data \n");
    return -1;
  }

  int *h_a, *d_a;
  float *h_b, *d_b;
  int img_size = image.rows * image.cols;

  h_a = (int *)malloc(img_size * sizeof(int));
  h_b = (float *)malloc(img_size * sizeof(float));

  cudaMalloc((void **) &d_a, img_size * sizeof(int));
  cudaMalloc((void **) &d_b, img_size * sizeof(float));

  cudaMemcpy(d_a, h_a, img_size * sizeof(int), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks((int)ceil((float)image.cols/threadsPerBlock.x), (int)ceil((float)image.rows/threadsPerBlock.y));

  gpuGrayScale<<<numBlocks, threadsPerBlock>>>(d_a, d_b, image.cols, image.rows);
  cout << "im here" << endl;
  cudaMemcpy(h_b, d_b, img_size, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);

  //namedWindow("Display Image", WINDOW_AUTOSIZE );
  //imshow("Display Image", image);
  
  //Mat img = (Mat_<float>(image.rows, image.cols) << h_b);
  //img = h_b;
  /*
  float r,g,b;
  for(int y=0;y<image.rows;y++){
    for(int x=0;x<image.cols;x++){
      // get pixel
      Vec3b color = img.at<Vec3b>(Point(x,y));

      r = color[0];
      g = color[1];
      b = color[2];

      //I = .299f * R + .587f * G + .114f * B
      color[2] = (r * 0.299 + g * 0.587 + b * 0.114);
      color[1] = (r * 0.299 + g * 0.587 + b * 0.114);
      color[0] = (r * 0.299 + g * 0.587 + b * 0.114);

      // set pixel
      img.at<Vec3b>(Point(x,y)) = color;
    }
  }
  */
  //imwrite("lena_out.jpg", img);

  //cout << h_b[0] << endl;

  //waitKey(0);

  return 0;
}