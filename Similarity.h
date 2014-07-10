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

        return total / count + count / 2.0;
    }

    // Elementwise disance of two images.
    int elementWiseDistance (Mat& mat1, Mat& mat2)
    {
        if (mat1.size() != mat2.size())
        {
            cout << "Error in elementWiseDistance! Matrices are not the same size";
            return -1;
        }

        int difference = 0;

        // cout << mat2 << endl;

        // cout << mat2.size() << " " << mat2.channels() << endl;

        for (int r = 0; r < mat1.rows; r++)
            for (int c = 0; c < mat1.cols; c++){
                // cout << (int) (mat2.at<Vec<uchar, 1> >(4*r,4*c))[0] << " ";
                difference += abs((int) (mat1.at<Vec<uchar, 1> >(r,c))[0] - (int) (mat2.at<Vec<uchar, 1> >(4*r,4*c))[0]);
 }  // cout << endl;
        // Calculate mean difference
        return difference / (mat1.rows * mat1.cols);
    }

    // Given two images, return a similarity score.
    int getSimilarity( Mat& mat1, Mat& mat2)
    {
        if (mat1.size() != mat2.size())
            return similarityOfDifferentSizedImages(mat1, mat2);

        int sim = elementWiseDistance(mat1, mat2);
        // sim += norm(mat1, mat2);//, NORM_RELATIVE_L2);

        return sim;
    }

    int compareIandFs(iandf if1, iandf if2)
    {
        int sim = 0;

        // sim += (int) compareDescriptors(if1.surfs, if2.surfs) * 10;
        // sim += (int) compareDescriptors(if1.sifts, if2.sifts) ;/// 3;
        sim += getSimilarity(if1.pixSum, if2.pixSum);
        // sim += getSimilarity(if1.bw, if2.bw);

        // cout
        // << "SURFS: " << 10 * (int) compareDescriptors(if1.surfs, if2.surfs)
        // << "\tSIFTS: " << (int) compareDescriptors(if1.sifts, if2.sifts) / 3
        // << "\tPXSUM: " << getSimilarity(if1.pixSum, if2.pixSum)
        // << "\tABOVE: " << getSimilarity(if1.bw, if2.bw) / 2
        // << "\tTOTAL: " << sim
        // << endl;

        // #define S "Match"
        // namedWindow(S);
        // imshow(S, if1.im);
        // waitKey(0);

        return sim;
    }
}