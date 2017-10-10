#include <stdio.h>
#include <time.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

__global__ void gpuGrayScale(unsigned char *imgIn, unsigned char *imgOut, int cols, int rows){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned char r,g,b;
  if((row < rows) && (col < cols)){
    r = imgIn[(row * cols + col) * 3 + 2];
    g = imgIn[(row * cols + col) * 3 + 1];
    b = imgIn[(row * cols + col) * 3 + 0];

    imgOut[row * cols + col] = r * 0.299 + g * 0.587 + b * 0.114;
  }
}


int main(int argc, char** argv )
{

  unsigned char *imageIn, *h_imageOut, *d_imageIn, *d_imageOut;
  
  Mat image;
  image = imread( argv[1], 1 );
  
  if ( argc != 2 )
  {
    printf("usage: DisplayImage <Image_Path>\n");
    return -1;
  }

  int cols = image.cols;
  int rows = image.rows;

  int imgInSize = sizeof(unsigned char) * cols * rows * image.channels();
  int imgOutSize = sizeof(unsigned char) * cols * rows;

  imageIn = (unsigned char*)malloc(imgInSize);
  h_imageOut = (unsigned char*)malloc(imgOutSize);

  cudaMalloc((void**)&d_imageIn, imgInSize);
  cudaMalloc((void**)&d_imageOut, imgOutSize);

  imageIn = image.data;

  cudaMemcpy(d_imageIn, imageIn, imgInSize, cudaMemcpyHostToDevice);

  int threads = 16;
  dim3 numThreads(threads, threads);
  dim3 blockDim((int)ceil((float)cols/threads), (int)ceil((float)rows/threads));

  gpuGrayScale<<<blockDim, numThreads>>>(d_imageIn, d_imageOut, cols, rows);
  cudaDeviceSynchronize();

  cudaMemcpy(h_imageOut, d_imageOut, imgOutSize, cudaMemcpyDeviceToHost);

  Mat imageOut;
  imageOut.create(rows, cols, CV_8UC1);
  imageOut.data = h_imageOut;

  cout << imageOut.channels() << endl << h_imageOut[0] << endl;

  imwrite("lena_out.jpg", imageOut);

  //waitKey(0);

  cudaFree(d_imageIn);
  cudaFree(d_imageOut);

  return 0;
}
