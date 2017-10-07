#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    Mat image, image_gray;
		Mat grad;
		//char* window_name = "Sobel Demo - Simple Edge Detector";
		int scale = 1;
		int delta = 0;
		int ddepth = CV_16S;

    image = imread( argv[1], 1 );

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    //namedWindow("Display Image", WINDOW_AUTOSIZE );

    cvtColor( image, image_gray, CV_BGR2GRAY );

    /// Generate grad_x and grad_y
		Mat grad_x, grad_y;
		Mat abs_grad_x, abs_grad_y;

		/// Gradient X
		//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
		Sobel( image_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
		convertScaleAbs( grad_x, abs_grad_x );

		/// Gradient Y
		//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
		Sobel( image_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
		convertScaleAbs( grad_y, abs_grad_y );

		/// Total Gradient (approximate)
		addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

    imwrite("lena_out.jpg", grad);

    //imshow("Display Image", grad);

    cout << image.size() << endl;

    //waitKey(0);

    return 0;
}
