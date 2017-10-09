cmake_minimum_required(VERSION 2.8)
project( DisplayImage )
find_package( OpenCV REQUIRED )
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( DisplayImage_cu DisplayImage.cu )
target_link_libraries( DisplayImage_cu ${OpenCV_LIBS} )


