cmake_minimum_required(VERSION 2.8)
project( openCV_GPU )
find_package( OpenCV REQUIRED )
add_executable( openCV_GPU openCV_GPU.cpp )
#find_package(CUDA REQUIRED)
#set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -arch=sm_52)
#include_directories( ${OpenCV_INCLUDE_DIRS} )
#cuda_add_executable( openCV_GPU openCV_GPU.cu )
target_link_libraries( openCV_GPU ${OpenCV_LIBS} )