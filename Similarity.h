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
        // First match descriptors
        FlannBasedMatcher matcher;
        vector<DMatch> matches;
        // matcher.match( desc1, desc2, matches );

        // double max_dist = 0; double min_dist = 100;

        // for( int i = 0; i < matches.size(); i++ )
        // { 
        //     double dist = matches[i].distance;
        //     if( dist < min_dist ) min_dist = dist;
        //     if( dist > max_dist ) max_dist = dist;
        // }

        double total = 0.0;
        double count = 0.0;

        // // Only look at the good ones
        // for( int i = 0; i < matches.size(); i++ )
        // {
        //     if( matches[i].distance <= max(2*min_dist, 0.02) )
        //     { 
        //         total += matches[i].distance;
        //         count += 1.0;
        //     }
        // }

        vector<vector<DMatch> > vecmatches;
        float ratio = 0.75;

        matcher.knnMatch(desc1, desc2, vecmatches, 2);
        for (int i = 0; i < vecmatches.size(); i++)
            if (vecmatches[i][0].distance < ratio * vecmatches[i][1].distance)
                matches.push_back(vecmatches[i][0]);

        for( int i = 0; i < matches.size(); i++ )
        {
            // if( matches[i].distance <= max(2*min_dist, 0.02) )
            // { 
                total += matches[i].distance;
                count += 1.0;
            // }
        }

        if (count < 2)
            return 1000.0;
        // cout << total / count << " ";
        return 1/count; // + count / 2.0;
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

        for (int r = 0; r < mat1.rows; r++)
            for (int c = 0; c < mat1.cols; c++)
                difference += abs((int) (mat1.at<Vec<uchar, 1> >(r,c))[0] - (int) (mat2.at<Vec<uchar, 1> >(4*r,4*c))[0]);
            // Total sketch with that (4*r, 4*c), but hey--if it works...

        // return mean difference
        return difference / (mat1.rows * mat1.cols);
    }

    // Given two images, return a similarity score.
    int getSimilarity( Mat& mat1, Mat& mat2)
    {
        if (mat1.size() != mat2.size())
            return similarityOfDifferentSizedImages(mat1, mat2);

        int sim = elementWiseDistance(mat1, mat2);
        // sim += norm(mat1, mat2);
        // ^ has issues with Mat types

        return sim;
    }

    int compareIandFs(iandf if1, iandf if2)
    {
        int sim = 0;

        // a linear combination of similarity tests:
        
        sim += compareDescriptors(if1.surfs, if2.surfs) * 100000.0;
        // sim += (int) compareDescriptors(if1.sifts, if2.sifts) ;/// 3;
        // sim += getSimilarity(if1.pixSum, if2.pixSum);
        // sim += getSimilarity(if1.bw, if2.bw);

        // cout
        // << "SURFS: " << 10 * (int) compareDescriptors(if1.surfs, if2.surfs)
        // << "\tSIFTS: " << (int) compareDescriptors(if1.sifts, if2.sifts) / 3
        // << "\tPXSUM: " << getSimilarity(if1.pixSum, if2.pixSum)
        // << "\tABOVE: " << getSimilarity(if1.bw, if2.bw) / 2
        // << "\tTOTAL: " << sim
        // << endl;

        return sim;
    }
}