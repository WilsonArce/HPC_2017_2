
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
  cudaError_t error = cudaSuccess;
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

  error = cudaMalloc((void**)&d_imageIn, imgInSize);
  if(error != cudaSuccess){
      printf("Error reservando memoria para d_imageIn\n -> %s\n", cudaGetErrorString(error));
      exit(-1);
  }
  cudaMalloc((void**)&d_imageOut, imgOutSize);

  imageIn = image.data;

  cudaMemcpy(d_imageIn, imageIn, imgInSize, cudaMemcpyHostToDevice);

  int threads = 32;
  dim3 numThreads(threads, threads);
  dim3 blockDim(ceil(cols/float(threads)), ceil(rows/float(threads)));

  gpuGrayScale<<<blockDim, numThreads>>>(d_imageIn, d_imageOut, cols, rows);
  cudaDeviceSynchronize();

  cudaMemcpy(h_imageOut, d_imageOut, imgOutSize, cudaMemcpyDeviceToHost);

  Mat imageOut;
  imageOut.create(rows, cols, CV_8UC1);
  imageOut.data = h_imageOut;

  cout << imageOut.channels() << endl << sizeof(d_imageOut) << endl;

  imwrite("lena_out.jpg", imageOut);

  //waitKey(0);

  cudaFree(d_imageIn);
  cudaFree(d_imageOut);

  return 0;
}
/*

//kala855 version

#include <cv.h>
#include <highgui.h>
#include <time.h>
#include <cuda.h>

#define RED 2
#define GREEN 1
#define BLUE 0

using namespace cv;


__global__ void img2gray(unsigned char *imageInput, int width, int height, unsigned char *imageOutput){
  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int col = blockIdx.x*blockDim.x+threadIdx.x;

  if((row < height) && (col < width)){
      imageOutput[row*width+col] = imageInput[(row*width+col)*3+RED]*0.299 + imageInput[(row*width+col)*3+GREEN]*0.587 \
                                    + imageInput[(row*width+col)*3+BLUE]*0.114;
  }
}


int main(int argc, char **argv){
  cudaError_t error = cudaSuccess;
  clock_t start, end, startGPU, endGPU;
  double cpu_time_used, gpu_time_used;
  char* imageName = argv[1];
  unsigned char *dataRawImage, *d_dataRawImage, *d_imageOutput, *h_imageOutput;
  Mat image;
  image = imread(imageName, 1);

  if(argc !=2 || !image.data){
      printf("No image Data \n");
      return -1;
  }

  Size s = image.size();

  int width = s.width;
  int height = s.height;
  int size = sizeof(unsigned char)*width*height*image.channels();
  int sizeGray = sizeof(unsigned char)*width*height;


  dataRawImage = (unsigned char*)malloc(size);
  error = cudaMalloc((void**)&d_dataRawImage,size);
  if(error != cudaSuccess){
      printf("Error reservando memoria para d_dataRawImage\n -> %s\n", cudaGetErrorString(error));
      exit(-1);
  }

  h_imageOutput = (unsigned char *)malloc(sizeGray);
  error = cudaMalloc((void**)&d_imageOutput,sizeGray);
  if(error != cudaSuccess){
      printf("Error reservando memoria para d_imageOutput\n");
      exit(-1);
  }


  dataRawImage = image.data;

  /*for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
          dataRawImage[(i*width+j)*3+BLUE] = 0;
      }
  }*/
/*
  startGPU = clock();
  error = cudaMemcpy(d_dataRawImage,dataRawImage,size, cudaMemcpyHostToDevice);
  if(error != cudaSuccess){
      printf("Error copiando los datos de dataRawImage a d_dataRawImage \n");
      exit(-1);
  }

  int blockSize = 32;
  dim3 dimBlock(blockSize,blockSize,1);
  dim3 dimGrid(ceil(width/float(blockSize)),ceil(height/float(blockSize)),1);
  img2gray<<<dimGrid,dimBlock>>>(d_dataRawImage,width,height,d_imageOutput);
  cudaDeviceSynchronize();
  cudaMemcpy(h_imageOutput,d_imageOutput,sizeGray,cudaMemcpyDeviceToHost);
  endGPU = clock();

  Mat gray_image;
  gray_image.create(height,width,CV_8UC1);
  gray_image.data = h_imageOutput;

  start = clock();
  Mat gray_image_opencv;
  cvtColor(image, gray_image_opencv, CV_BGR2GRAY);
  end = clock();


  imwrite("./Gray_Image.jpg",gray_image);

  namedWindow(imageName, WINDOW_NORMAL);
  namedWindow("Gray Image CUDA", WINDOW_NORMAL);
  namedWindow("Gray Image OpenCV", WINDOW_NORMAL);

  imshow(imageName,image);
  imshow("Gray Image CUDA", gray_image);
  imshow("Gray Image OpenCV",gray_image_opencv);

  waitKey(0);

  //free(dataRawImage);
  gpu_time_used = ((double) (endGPU - startGPU)) / CLOCKS_PER_SEC;
  printf("Tiempo Algoritmo Paralelo: %.10f\n",gpu_time_used);
  cpu_time_used = ((double) (end - start)) /CLOCKS_PER_SEC;
  printf("Tiempo Algoritmo OpenCV: %.10f\n",cpu_time_used);
  printf("La aceleración obtenida es de %.10fX\n",cpu_time_used/gpu_time_used);

  cudaFree(d_dataRawImage);
  cudaFree(d_imageOutput);
  return 0;
}
*/