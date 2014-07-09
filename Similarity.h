// File: Similarity.h

#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"

using namespace cv;
using namespace std;


namespace similarities 
{

    // Image and Features structure for each point and direction
    struct iandf{
        Mat im;
        Mat surfs;
        Mat sifts;
        Mat bw;
        Mat pixSum;
        int sim;
    };
    // given two images of different size, return a similarity score
    int similarityOfDifferentSizedImages(const Mat& mat1, const Mat& mat2)
    {
    // TODO
        return 10000;
    }

    // given two sets of keypoints and descriptors, return a similarity score
    float compareDescriptors(const Mat& desc1, const Mat& desc2)
    {
        FlannBasedMatcher matcher;
        vector<DMatch> matches;
        matcher.match( desc1, desc2, matches );

        double max_dist = 0; double min_dist = 100;

        for( int i = 0; i < matches.size(); i++ )
        { 
            double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

        double total = 0.0;
        double count = 0.0;

        for( int i = 0; i < matches.size(); i++ )
        {
            if( matches[i].distance <= max(2*min_dist, 0.02) )
            { 
                total += matches[i].distance;
                count += 1.0;
            }
        }

        if (count == 0.0)
            return 1000.0;

        return total / count + count / 2;
    }

    // Elementwise disance of two images.
    int elementWiseDistance (Mat& mat1, Mat& mat2)
    {
        if (mat1.size() != mat2.size())
        {
            cout << "Error in elementWiseDistance! Matrices are not the same size";
            return -1;
        } 


        namedWindow("1");
        imshow("1", mat1);
        waitKey(0);

        int difference = 0;

        // cout << mat1 << endl;

        for (int r = 0; r < mat1.rows; r++)
        {
            for (int c = 0; c < mat1.cols; c++)
            {
                string t = "\t";
                // Vec3b intensity = mat1.at<Vec3b>(r, c);
                // cout << intensity.val[0] << "\t" << intensity.val[1] << "\t" << intensity.val[2] << endl;
                // cout << mat1.type();// << endl;
                // cout << "double" << mat1.at<double>(r,c) << endl;
                // cout << "float" << mat1.at<float>(r,c) << endl;
                // cout << "short int" << mat1.at<short int>(r,c) << endl;
                // cout << "ushort int" << mat1.at<unsigned short int>(r,c) << endl;
                // cout << "int" << mat1.at<int>(r,c) << endl;
                // cout << "long int" << mat1.at<long int>(r,c) << endl;
                // cout << "unsigned long int" << mat1.at<unsigned long int>(r,c) << endl;
                // cout << "unsigned int" << mat1.at<unsigned int>(r,c) << endl;
                // cout << "sizet" << mat1.at<size_t>(r,c) << endl;

                difference += abs(mat1.at<int>(r,c) - mat2.at<int>(r,c));
            }
        }

        // Calculate mean difference
        return difference / (mat1.rows * mat1.cols);
    }

    // Given two images, return a similarity score.
    int getSimilarity( Mat& mat1, Mat& mat2)
    {
        // cout << "BLAH!" << endl;
        if (mat1.size() != mat2.size())
            return similarityOfDifferentSizedImages(mat1, mat2);

        int ewd = elementWiseDistance(mat1, mat2);

        cout << "ewd" << ewd << endl; // << "\tmnorm "<< matrixnorm << endl;

        return ewd ;//+ 0.001 * matrixnorm;
    }

    int compareIandFs(iandf if1, iandf if2)
    {
        cout << if1.pixSum.channels() << endl;
        cout << if2.pixSum.channels() <<endl;       
        cout << if1.bw.channels() << endl;
        cout << if2.bw.channels() <<endl;

        int sim = 0;

        sim += 10 * (int) compareDescriptors(if1.surfs, if2.surfs);
        sim += (int) compareDescriptors(if1.sifts, if2.sifts) / 3;
        sim += getSimilarity(if1.pixSum, if2.pixSum);
        // cout << getSimilarity(if1.pixSum, if2.pixSum) << "\t" << getSimilarity(if1.bw, if2.bw) << endl;
        sim += getSimilarity(if1.bw, if2.bw) / 2;

        return sim;
    }
}