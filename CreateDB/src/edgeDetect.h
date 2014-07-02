#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "cvaux.h"

#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

namespace Edges 
{
    Mat getSobelEdges(const Mat& src, const int scale=1, const int delta=0, const int ddepth=CV_16S)
    {
        Mat src_gray;
        Mat grad, grad_x, grad_y;
        Mat abs_grad_x, abs_grad_y;

        GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

        cvtColor( src, src_gray, CV_RGB2GRAY );

        Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_x, abs_grad_x );

        Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_y, abs_grad_y );

        addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

        return grad;
    }

    Mat getScharrEdges(const Mat& src, const int scale=1, const int delta=0, const int ddepth=CV_16S)
    {
        Mat src_gray;
        Mat grad, grad_x, grad_y;
        Mat abs_grad_x, abs_grad_y;

        GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

        cvtColor( src, src_gray, CV_RGB2GRAY );

        Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_x, abs_grad_x );

        Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_y, abs_grad_y );

        addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

        return grad;
    }
}