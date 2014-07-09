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
        float sim;
    };
    // given two images of different size, return a similarity score
    int similarityOfDifferentSizedImages(const Mat& mat1, const Mat& mat2)
    {
    // TODO
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
    float elementWiseDistance (const Mat& mat1, const Mat& mat2)
    {
        if (mat1.size() != mat2.size())
        {
            cout << "Error in elementWiseDistance! Matrices are not the same size";
            return -1;
        } 

        int difference = 0;

        for (int r = 0; r < mat1.rows; r++)
        {
            for (int c = 0; c < mat1.cols; c++)
            {
                difference += abs(mat1.at<int>(r,c) - mat2.at<int>(r,c));
            }
        }

        // Calculate mean difference
        return difference/(mat1.rows * mat1.cols);
    }

    // Given two images, return a similarity score.
    int getSimilarity(const Mat& mat1, const Mat& mat2)
    {
        if (mat1.size() != mat2.size())
            return similarityOfDifferentSizedImages(mat1, mat2);

        float ewd = elementWiseDistance(mat1, mat2);
        float matrixnorm = norm(mat1, mat2);

        return (int) ewd + 0.001 * matrixnorm;
    }

    float compareIandFs(iandf if1, iandf if2)
    {
        float sim = 0.0;
        // cout << if2.pixSum << endl;
        sim += 10*compareDescriptors(if1.surfs, if2.surfs);
        sim += 0.3*compareDescriptors(if1.sifts, if2.sifts);
        sim += 0*getSimilarity(if1.pixSum, if2.pixSum);
        sim += 0*getSimilarity(if1.bw, if2.bw);

        return sim;
    }
}