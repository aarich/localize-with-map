#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "cvaux.h"

#include <pcl/io/pcd_io.h>

#include "boost/filesystem.hpp"   

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>

#include "../../averageImage.h"
#include "../../Similarity.h"

using namespace cv;
using namespace std;
namespace fs = ::boost::filesystem;

// Print usage
static void printPrompt( const string& applName )
{
    cout << "/*\n"
    << " * Given a folder of photos, this computes descriptors and info of image.\n"
         << " */\n" << endl;

    cout << endl << "Format:\n" << endl;
    cout << applName << " InputsImages.txt" << endl;
    cout << endl;
}

// return the filenames of all files that have the specified extension
// in the specified directory and all subdirectories
void get_all(const fs::path& root, vector<fs::path>& ret)
{  
    if (!fs::exists(root)) return;

    if (fs::is_directory(root))
    {
        fs::recursive_directory_iterator it(root);
        fs::recursive_directory_iterator endit;
        while(it != endit)
        {
            if (fs::is_regular_file(*it))
            {
                ret.push_back(it->path().filename());
            }
            ++it;
        }
    }
}

int main(int argc, char** argv)
{
    TickMeter tm;
    tm.reset();
    tm.start();
    if( argc < 2 )
    {
        printPrompt( argv[0] );
        return -1;
    }

    initModule_nonfree();

    // Get Input Data
    ifstream file(argv[1]);
    if ( !file.is_open() )
        return false;
    
    string str;

        // Number of divisions
    getline( file, str ); getline( file, str );
    float divs = atoi(str.c_str());
        // Directory to look for photos
    getline( file, str ); getline( file, str );
    string dir =str.c_str();
        // Directory to save BW images (abov ebelwo)
    getline( file, str ); getline( file, str );
    string bwdir =str.c_str();
        // Directory to save GS images (avg pixel sum)
    getline( file, str ); getline( file, str );
    string gsdir =str.c_str();
        // Directory to save descriptors
    getline( file, str ); getline( file, str );
    string kdir =str.c_str();

    file.close();
    // Done Getting Input Data

    map<vector<float>, Mat> imagemap;

    vector<KeyPoint> Keypoints;
    Mat Descriptors;

    int minHessian = 300;

    SurfFeatureDetector SurfDetector (minHessian);
    SiftFeatureDetector SiftDetector (minHessian);

    SurfDescriptorExtractor SurfExtractor;
    SiftDescriptorExtractor SiftExtractor;
    // Load Images

    // First look into the folder to get a list of filenames
    vector<fs::path> ret;
    const char * pstr = dir.c_str();
    fs::path p(pstr);
    get_all(pstr, ret);

    for (int i = 0; i < ret.size(); i++)
    {
            // Load Image via filename
        string fn = ret[i].string();
        istringstream iss(fn);
        vector<string> tokens;
        copy(istream_iterator<string>(iss), istream_iterator<string>(), back_inserter<vector<string> >(tokens));

            // Construct ID from filename
        vector<float> ID;
        for (int i = 0; i < 6; i++) // 6 because there are three location floats and three direction floats
            ID.push_back(::atof(tokens[i].c_str()));
        string imfn = dir + "/" + fn;

            // Read image and add to imagemap.
        Mat m = imread(imfn);
        imagemap[ID] = m;
    }
    tm.stop();
    float load = tm.getTimeSec();
    tm.reset();
    tm.start();

    for (map<vector<float>, Mat>::iterator i = imagemap.begin(); i != imagemap.end(); ++i)
    {
        // Create image name and storagename
        string imfn = "/";
        string kpfn = "/";
        for (int j = 0; j < 6; j++)
        {
            imfn += boost::to_string(i->first[j]) + " ";
            kpfn += boost::to_string(i->first[j]) + " ";
        }
        imfn += ".jpg";
        kpfn += ".yml";

        FileStorage store(kdir + kpfn, cv::FileStorage::WRITE);

        SurfDetector.detect(i->second, Keypoints);
        SurfExtractor.compute(i->second, Keypoints, Descriptors);
        write(store,"SurfDescriptors",Descriptors);

        SiftDetector.detect(i->second, Keypoints);
        SiftExtractor.compute(i->second, Keypoints, Descriptors);
        write(store,"SiftDescriptors",Descriptors);

        store.release();

        Mat gs = averageImage::getPixSumFromImage(i->second, divs);

        imwrite(gsdir + imfn, gs);
        imwrite(bwdir + imfn, averageImage::aboveBelow(gs));
    }
    tm.stop();
    float analysis = tm.getTimeSec();

    cout << ">\n"
        << "Loading took " << load << " seconds for " << imagemap.size() << " images (" 
        << (int) imagemap.size()/load << " images per second)." << endl;
    cout << "Analysis took " << analysis << " seconds (" << (int) imagemap.size()/analysis << " images per second)." << endl; 
        

return 0;
}