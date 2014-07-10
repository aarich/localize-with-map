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

void removeBad(vector<KeyPoint> kps, Mat& img)
{
    vector<KeyPoint>::iterator iter;
    for (iter = kps.begin(); iter != kps.end(); ) 
    {
        Vec3b v = img.at<Vec3b>(iter->pt);
        if (v == Vec3b(255, 0, 255))
            iter = kps.erase(iter);
        else
            ++iter;
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
        // Image extension
    getline( file, str ); getline( file, str );
    string extension =str.c_str();
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

    cout << "Loading Images" << endl;

    vector<KeyPoint> Keypoints;
    Mat Descriptors;

    int minHessian = 400;

    SurfFeatureDetector SurfDetector (3000, 6, 2, true, true);
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
        // cout << fn << endl;
        istringstream iss(fn);
        vector<string> tokens;
        // copy(istream_iterator<string>(iss), istream_iterator<string>(), back_inserter<vector<string> >(tokens));
        string delimiter = "_";
        if (fn[0] != delimiter.c_str()[0]){
            cout << "\033[1;33mExtraneous file found: " << fn << "\033[0m" << endl;
            continue;
        }
        fn = fn.substr(1,fn.length());
        size_t pos = 0;
        string token;
        while ((pos = fn.find(delimiter)) != std::string::npos) 
        {
            token = fn.substr(0, pos);
            fn.erase(0, pos + delimiter.length());
            tokens.push_back(token);
        }
        fn = ret[i].string();

            // Construct ID from filename
        vector<float> ID;
        for (int j = 0; j < 6; j++) // 6 because there are three location floats and three direction floats
            ID.push_back(::atof(tokens[j].c_str()));
        string imfn = dir + "/" + fn;

            // Read image and add to imagemap.
                // cout << i << ": " << imfn << endl;

        Mat m = imread(imfn);
        imagemap[ID] = m;
    }
    tm.stop();
    float load = tm.getTimeSec();
    tm.reset();
    tm.start();

    int count = 0;
    int total = imagemap.size();

    cout << total << " images found.\nComputing keypoints and coarse images." << endl;

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
        imfn += extension;
        kpfn += ".yml";

        FileStorage store(kdir + kpfn, cv::FileStorage::WRITE);

        SurfDetector.detect(i->second, Keypoints);
        removeBad(Keypoints, i->second);
        SurfExtractor.compute(i->second, Keypoints, Descriptors);
        write(store,"SurfDescriptors",Descriptors);

        SiftDetector.detect(i->second, Keypoints);
        removeBad(Keypoints, i->second);
        SiftExtractor.compute(i->second, Keypoints, Descriptors);
        write(store,"SiftDescriptors",Descriptors);

        store.release();

        Mat gs = averageImage::getPixSumFromImage(i->second, divs);

        imwrite(gsdir + imfn, gs);
        imwrite(bwdir + imfn, averageImage::aboveBelow(gs));

        tm.stop();
        double s = tm.getTimeSec();
        double x = s * (double) total / (double) count;
        tm.start();

        count++;

        if (count%50 == 0){
            cout << 100 * count / total << " percent done. Estimated Time Remaining: " << (x-s)/60.0 << " minutes." << endl;
        }
    }

    tm.stop();
    float analysis = tm.getTimeSec();

    cout << ">\n"
    << "Loading took " << load << " seconds for " << imagemap.size() << " images (" 
        << (int) imagemap.size()/load << " images per second)." << endl;
cout << "Analysis took " << analysis << " seconds (" << (int) imagemap.size()/analysis << " images per second)." << endl; 


return 0;
}