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

namespace averageImage {

    // Get the sum of the pixel values within a square of an image
    #define T double
    T getMagnitude(const int leftx, const int uppery, const int rightx, const int lowery, const Mat& img)
    {
        double area = (double) ((rightx - leftx) * (lowery - uppery));
        T one = (T) img.at<T>(uppery, leftx) / area;//(leftx, uppery);
        T two = (T) img.at<T>(uppery, rightx) / area;//(rightx, uppery);
        T three = (T) img.at<T>(lowery, leftx) / area;//(leftx, lowery);
        T four = (T) img.at<T>(lowery,rightx) / area;//(rightx, lowery);
        // 1 -- 2
        // |    |
        // 3 -- 4
        // cout << "Values: " << one << "\t" << two << "\t" << three << "\t" << four << endl;
        // cout << img.at<T>(uppery, leftx);
        // cout << four + one - two - three  << "\t";
        return four + one - two - three;
    }

    // Get the pixel sum image of an integral picture.
    Mat getPixSum(const Mat& image, const int divisions)
    {
        Mat results(divisions, divisions, CV_64F);

        float h_division = (float)(image.rows-1)/ (float)divisions;
        float w_division = (float)(image.cols-1)/ (float)divisions;

        int uppery, lowery, leftx, rightx;
        double mag;

        for (int r = 0; r < divisions; r++)
        {
            uppery = (int) r*h_division;
            lowery = (int) (r+1)*h_division;
            for (int c = 0; c < divisions; c++)
            {
                leftx = (int) c*w_division;
                rightx = (int) (c + 1) * w_division;
                // cout << "r: " << r << "\t" << "c: " << c << "\t" << leftx << "\t" << rightx << "\t" << uppery << "\t" << lowery << endl;
                mag = getMagnitude(leftx, uppery, rightx, lowery, image);
                // cout << "\tmag: " << mag << endl;

                results.at<double>(r, c) = mag;
            }
        }
        // cout << "Results:" << endl << results << endl;
        normalize(results, results, 0, 255, NORM_MINMAX, CV_64F);
        
        Mat results2(results.size(), CV_32S);

        for (int r = 0; r < results.rows; r++)
        {
            for (int c = 0; c < results.cols; c++)
            {
                results2.at<int>(r,c) = (int)results.at<double>(r,c);
            }
        }

        // cout << "Normalized int Results:\n" << results2 << endl;

        return results2;
    }

    // Get the bw above and below image of a picture.
    // Input is the average pixel image
    // Output is the bw image with above average colored white
    // and below painted black.
    Mat aboveBelow(const Mat& image)
    {
        Mat results(image.size(), CV_32S);
        int avg = (int) (mean(image)).val[0];

        for (int r = 0; r < image.rows; r++)
        {
            for (int c = 0; c < image.cols; c++)
            {
                int i = image.at<int>(r,c);
                if (i > avg)
                    results.at<int>(r,c) = 255;
                else
                    results.at<int>(r,c) = 0;
            }
        }

        return results;
    }

    Mat getPixSumFromImage(const Mat& image, const int divisions)    
    {
        Mat newImage;
        if (image.channels() > 1)
        {
            cvtColor(image, newImage, CV_BGR2GRAY);
            cout << newImage.channels() << "\t" << newImage.size() << endl;
        }
        else
            newImage = image;

        Mat iImage;
        integral(image, iImage, CV_64F);
        
        Rect ROI (1, 1, iImage.cols-1, iImage.rows-1);
        Mat cropped = iImage(ROI);

        return getPixSum(cropped, divisions);
    }

}