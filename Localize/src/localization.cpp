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
    cout 
    << "/*\n"
    << " * Must have a folder for images and for keypoints and descriptors.\n"
    << " * Filenames must follow: \"x y z anglex angley angle z .jpg||.yaml\"\n"
    << " */\n" << endl;

    cout << endl << "Format:\n" << endl;
    cout << applName << " [Inputs.txt]" << endl;
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
    if( argc < 2 )
    {
        printPrompt( argv[0] );
        return -1;
    }

    // initModule_nonfree();

    // **************
    // Get Input Data
    ifstream file(argv[1]);
    if ( !file.is_open() )
        return false;
    
    string str;
    
        // Image Name
    getline( file, str ); getline( file, str );
    string image_name = str;
        // Number of divisions
    getline( file, str ); getline( file, str );
    float divs = atoi(str.c_str());
        // Number of images to return
    getline( file, str ); getline( file, str );
    int numtoreturn = atoi(str.c_str());    
        // Directory to look for photos
    getline( file, str ); getline( file, str );
    string dir =str.c_str();
        // Directory to look for bw photos
    getline( file, str ); getline( file, str );
    string bwdir =str.c_str();
        // Directory to look for gs photos
    getline( file, str ); getline( file, str );
    string gsdir =str.c_str();
        // Directory to look for kp and descriptors
    getline( file, str ); getline( file, str );
    string kdir =str.c_str();
    
    file.close();
    // Done with Input
    // ***************

    // A map to store all the rendered images and associated info
    map<vector<float>, similarities::iandf> imagemap;

    // iandf contains: im, surfs, sifts, bw, pixSum, sim
    // (Mat im, Mat surfs, Mat sifts, Mat bw, Mat pixSum, float sim)
    
    // tmp Descriptors
    Mat descriptors;

    // Now go into folder and get all available photos and features.    
    vector<fs::path> ret;
    const char * pstr = dir.c_str();
    fs::path p(pstr);

    // Get a list of all filenames
    get_all(pstr, ret);

    for (int i = 0; i < ret.size(); i++)
    {
        // Load Image via filename
        string fn = ret[i].string();
        istringstream iss(fn);
        vector<string> tokens;
        // Split up string.
        copy(istream_iterator<string>(iss), istream_iterator<string>(), back_inserter<vector<string> >(tokens));

        // Construct ID from filename tokens
        vector<float> ID;
        for (int j = 0; j < 6; j++) // 6 because there are three location floats and three direction floats
            ID.push_back(atof(tokens[j].c_str()));
        
        // Read image and add to imagemap.
        string imfn = dir + "/" + fn;
        similarities::iandf tmp;
        tmp.im = imread(imfn);
        imagemap[ID] = tmp;

        // Create Filename for loading descriptors
        string kpfn = kdir + "/";
        for (int j = 0; j < ID.size(); j++)
        {
            kpfn += boost::to_string(ID[j]) + " ";
        }

        kpfn = kpfn+ ".yml";

        // Create filestorage item to read from and add to map.
        FileStorage store(kpfn, FileStorage::READ);

        FileNode n1 = store["SurfDescriptors"];
        read(n1, descriptors);
        imagemap[ID].surfs = descriptors;

        FileNode n2 = store["SiftDescriptors"];
        read(n2, descriptors);
        imagemap[ID].sifts = descriptors;

        store.release();

        imagemap[ID].pixSum = imread(gsdir + "/" + fn, CV_LOAD_IMAGE_GRAYSCALE);
        imagemap[ID].bw = imread(bwdir + "/" + fn, CV_LOAD_IMAGE_GRAYSCALE);
        cout << imagemap[ID].pixSum.row(0) << endl;
        cout << imagemap[ID].pixSum.at<int>(0,0);
    }

    cout << "<\n  Analyzing Images ..." << endl;

    Mat image = imread(image_name);
    similarities::iandf matchingImage;
    matchingImage.im = image;

    if(! image.data ) // Check for invalid input
    {
        cout <<  "\033[1;31mCould not open or find the image\033[0m" << endl ;
        return -1;
    }

    map<vector<float>, similarities::iandf>::iterator item = imagemap.begin();
    int r = item->second.bw.rows;
    if (divs != r)
    {
        cout <<  "\033[1;33mRequested " << divs << " divs but saved images have " << r << " divs.\033[0m" << endl;
        divs = r;
    }

    matchingImage.pixSum = averageImage::getPixSumFromImage(image, divs);
    matchingImage.bw = averageImage::aboveBelow(matchingImage.pixSum);

    cout << matchingImage.pixSum.channels();

    // cout << gsimage <<endl;
    // imwrite("GS.png", gsimage);
    // namedWindow("GSIMAGE (Line 319)");
    // imshow("GSIMAGE (Line 319)", gsimage);
    // waitKey(0);
    // destroyWindow("GSIMAGE (Line 319)");

    vector<KeyPoint> imgKeypoints;
    Mat imgDescriptors;

    int minHessian = 300;

    SurfFeatureDetector SurfDetector (minHessian);
    SurfDescriptorExtractor SurfExtractor;
    SurfDetector.detect(image, imgKeypoints);
    SurfExtractor.compute(image, imgKeypoints, imgDescriptors);
    matchingImage.surfs = imgDescriptors;

    SiftDescriptorExtractor SiftExtractor;
    SiftFeatureDetector SiftDetector (minHessian);
    SiftDetector.detect(image, imgKeypoints);
    SiftExtractor.compute(image, imgKeypoints, imgDescriptors);
    matchingImage.sifts = imgDescriptors;

    TickMeter tm;
    tm.reset();
    tm.start();

    cout << ">\n<\n  Comparing Images ..." << endl;

    for (map<vector<float>, similarities::iandf>::iterator i = imagemap.begin(); i != imagemap.end(); ++i)
    {
        vector<float> ID = i->first;
        imagemap[ID].sim = similarities::compareIandFs(imagemap[ID], matchingImage);
    }

    map<vector<float>, int> top;
    bool gotone = false;

    typedef map<vector<float>, int>::iterator iter;

    // Choose the best ones!
    for (map<vector<float>, similarities::iandf>::iterator i = imagemap.begin(); i != imagemap.end(); ++i)
    {
        vector<float> ID = i->first;

        int sim = imagemap[ID].sim;

        if (!gotone)
        {
            top[ID] = sim;
            gotone = true;
        }

        iter it = top.begin();
        iter end = top.end();
        int max_value = it->second;
        vector<float> max_ID = it->first;
        for( ; it != end; ++it) 
        {
            int current = it->second;
            if(current > max_value) 
            {
                max_value = it->second;
                max_ID = it->first;
            }
        }

        // cout << "Sim: " << sim << "\tmax_value: " << max_value << endl;
        if (top.size() < numtoreturn)
            top[ID] = sim;
        else
        {
            if (sim < max_value)
            {
                top[ID] = sim;
                top.erase(max_ID);
            }
        }

    }
    tm.stop();
    double s = tm.getTimeSec();

    cout << ">\n<\n  Writing top " << numtoreturn << " images ..." << endl;

    int count = 1;
    namedWindow("Image");
    namedWindow("Match");
    namedWindow("ImageBW");
    namedWindow("MatchBW");
    namedWindow("ImageGS");
    namedWindow("MatchGS");

    imshow("Image", matchingImage.im);
    imshow("ImageBW", matchingImage.bw);
    imshow("ImageGS", matchingImage.pixSum);

    // cout << matchingImage.bw;

    for (iter i = top.begin(); i != top.end(); ++i)
    {
        vector<float> ID = i->first;

        cout << "  Score: "<< i->second << endl;
        string fn = "Sim_" + boost::to_string(count) + "_" + boost::to_string(i->second) + ".png";
        // imwrite(fn, imagemap[ID].im);
        count++;

        // normalize(imagemap[ID].bw, imagemap[ID].bw, 0, 255, NORM_MINMAX, CV_64F);
        // normalize(imagemap[ID].pixSum, imagemap[ID].pixSum, 0, 255, NORM_MINMAX, CV_64F);

        imshow("MatchBW", imagemap[ID].bw);
        waitKey(1);
        imshow("Match", imagemap[ID].im);
        waitKey(1);
        imshow("MatchGS", imagemap[ID].pixSum);

        waitKey(10000);
    }

    cout << ">\nComparisons took " << s << " seconds for " << imagemap.size() << " images (" 
        << (int) imagemap.size()/s << " images per second)." << endl;

return 0;
}