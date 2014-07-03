// File: Similarity.h

namespace similarities 
{
    // given two images of different size, return a similarity score
    int similarityOfDifferentSizedImages(const Mat& mat1, const Mat& mat2)
    {
    // TODO
    }

    // given two sets of keypoints and descriptors, return a similarity score
    int compareDescriptors()
    {
        // TODO
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

        /*
            Tools Available:
            norm
            element-wise comparison (distance)
        */
        float ewd = elementWiseDistance(mat1, mat2);
        float matrixnorm = norm(mat1, mat2);

        return (int) ewd + 0.01 * matrixnorm;
    }
}