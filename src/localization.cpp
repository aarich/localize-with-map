#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "cvaux.h"

#include <iostream>
#include <fstream>

#include "~/Desktop/Coarse Features/AverageImage/src/averageImage.h"

using namespace cv;
using namespace std;

// Print usage
static void printPrompt( const string& applName )
{
    cout << "/*\n"
         << " * Sandbox for doing LBP and and Casscade Classifier\n"
         << " */\n" << endl;

    cout << endl << "Format:\n" << endl;
    cout << "./" << applName << " [ImageName] [FileWithMoreImages(.txt)] [Divisions]" << endl;
    cout << endl;
}

static void readTrainFilenames( const string& filename, string& dirName, vector<string>& trainFilenames )
{
    trainFilenames.clear();

    ifstream file( filename.c_str() );
    if ( !file.is_open() )
        return;

    size_t pos = filename.rfind('\\');
    char dlmtr = '\\';
    if (pos == string::npos)
    {
        pos = filename.rfind('/');
        dlmtr = '/';
    }
    dirName = pos == string::npos ? "" : filename.substr(0, pos) + dlmtr;

    while( !file.eof() )
    {
        string str; getline( file, str );
        if( str.empty() ) break;
        trainFilenames.push_back(str);
    }
    file.close();
}

static bool readImages( const string& queryImageName, const string& trainFilename,
                 Mat& queryImage, vector <Mat>& trainImages, vector<string>& trainImageNames )
{
    cout << "< Reading the images..." << endl;
    queryImage = imread( queryImageName, CV_LOAD_IMAGE_GRAYSCALE);
    if( queryImage.empty() )
    {
        cout << "Query image can not be read." << endl << ">" << endl;
        return false;
    }
    string trainDirName;
    readTrainFilenames( trainFilename, trainDirName, trainImageNames );
    if( trainImageNames.empty() )
    {
        cout << "matching image filenames can not be read." << endl << ">" << endl;
        return false;
    }
    int readImageCount = 0;
    for( size_t i = 0; i < trainImageNames.size(); i++ )
    {
        string filename = trainDirName + trainImageNames[i];
        Mat img = imread( filename, CV_LOAD_IMAGE_GRAYSCALE );
        if( img.empty() )
            cout << "image " << filename << " can not be read." << endl;
        else
            readImageCount++;
        trainImages.push_back( img );
    }
    if( !readImageCount )
    {
        cout << "All images can not be read." << endl << ">" << endl;
        return false;
    }
    else
        cout << readImageCount << " matching images were read." << endl;
    cout << ">" << endl;

    return true;
}


int main(int argc, char** argv)
{
    if( argc < 4 )
    {
        printPrompt( argv[0] );
        return -1;
    }

    cv::initModule_nonfree();

    string queryImageName = argv[1];
    string fileWithTrainImages = argv[2];
    int divs = atoi(argv[3]);

    Mat queryImage;
    vector<Mat> matchingImages;
    vector<string> matchingImageNames;
    if( !readImages( queryImageName, fileWithTrainImages, queryImage, matchingImages, matchingImageNames) )
    {
        printPrompt( argv[0] );
        return -1;
    }

    // cout << "< Computing integrals \n";

    Mat iImage1;
    integral(queryImage, iImage1, CV_64F);

    vector<Mat> imatchingImages;
    for(size_t i = 0; i < matchingImages.size(); i++)
    {
        Mat img;
        integral(matchingImages[i], img, CV_64F);
        imatchingImages.push_back(img);
    }
    // cout << "> \n< Computing Pixel Sums\n";

    Rect ROI (1, 1, iImage1.cols-1, iImage1.rows-1);
    Mat cropped = iImage1(ROI);
    Mat pixsum1 = getPixSum(cropped, divs);

    imwrite("query.jpg", pixsum1);

    vector<Mat> pixsumMatchingImages;
    for(size_t i = 0; i < matchingImages.size(); i++)
    {
        Mat current = imatchingImages[i];
        Rect ROI (1, 1, current.cols-1, current.rows-1);
        Mat cropped = current(ROI);
        Mat pixsum;
        pixsum = getPixSum(current, divs);
        pixsumMatchingImages.push_back(pixsum);
        string fn = "sum_" + matchingImageNames[i];
        imwrite(fn, pixsum);
    }

    // cout << ">\n< Computing simularity\n";

    float best = 500;
    string bestnm;

    TickMeter tm;
    tm.start();

    // Calculate simularity
    for(size_t i = 0; i < matchingImages.size(); i++)
    {
        float sim = determineSimilarity(pixsum1, pixsumMatchingImages[i]);
        cout << matchingImageNames[i] << " similarity: " << (int) sim << endl;
        if (sim < best)
        {
            best = sim;
            bestnm = matchingImageNames[i];
        }
    }

    tm.stop();
    double matchTime = tm.getTimeMilli();

    cout << endl << bestnm << " was the closest match at " << divs << " divisions.\n\n";

    cout << "Matching time: " << matchTime << "ms. Per Image: " << matchTime/matchingImageNames.size() << " ms." << endl;

    return 0;
}
