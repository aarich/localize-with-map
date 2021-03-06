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
    << " * Must have four folders: See Inputs.txt for descriptions.\n"
    << " * Filenames must follow: \"x y z anglex angley angle z .jpg||.yaml,\" for any delimiter (declared in Inputs.txt)\n"
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
    // ^ Probably isn't necessary

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
        // File extension (jpg or png...)
    getline( file, str ); getline( file, str );
    string extension =str.c_str();
        // Delimiter in Filenames
    getline( file, str ); getline( file, str );
    string delimiter =str.c_str();
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

    // Now go into folder and get all available photos and features.    
    vector<fs::path> ret;
    const char * pstr = dir.c_str();
    fs::path p(pstr);
    // Get a list of all filenames
    get_all(pstr, ret);

    cout << "<\n  Attempting to load " << ret.size() << " images," << endl;
    cout << "  Then compare to " << image_name << endl;

    // tmp Descriptors
    Mat descriptors;

    for (int i = 0; i < ret.size(); i++)
    { // Load image, bw, gs, and descriptors via filename. 

        // fn is filename
        string fn = ret[i].string();
        // tokens is vector of strings in filename
        vector<string> tokens; 

        if (fn[0] != delimiter.c_str()[0]){
            cout << "\033[1;33m  Extraneous file found: " << fn << "\033[0m" << endl; // ]]
            continue;
        }

        // Split up string.
        fn = fn.substr(1, fn.length());
        size_t pos = 0;
        string token;
        while ((pos = fn.find(delimiter)) != std::string::npos) 
        {
            token = fn.substr(0, pos);
            fn.erase(0, pos + delimiter.length());
            tokens.push_back(token);
        }
        // Reset Filename
        fn = ret[i].string();

        // Construct ID from filename tokens
        vector<float> ID;
        for (int j = 0; j < 6; j++) // 6 because there are three location floats and three direction floats
            ID.push_back(atof(tokens[j].c_str()));
        
        // Read image and add to imagemap.

        // imfn is the image filename including directory
        string imfn = dir + "/" + fn;

        // Create a temporary image and features struct.
        similarities::iandf tmp;
        tmp.im = imread(imfn);

        // Add it to our map
        imagemap[ID] = tmp;

        // Create filename for loading descriptors
        string kpfn = kdir + "/" + delimiter;
        fn = delimiter;
        for (int j = 0; j < ID.size(); j++)
        {
            kpfn += boost::to_string(ID[j]) + delimiter;
            fn += boost::to_string(ID[j]) + delimiter;
        }

        fn += ".jpg";
        kpfn += ".yml";

        // Create filestorage item to read from and add to map.
        FileStorage store(kpfn, FileStorage::READ);

        FileNode n1 = store["SurfDescriptors"];
        read(n1, descriptors);
        imagemap[ID].surfs = descriptors;

        FileNode n2 = store["SiftDescriptors"];
        read(n2, descriptors);
        imagemap[ID].sifts = descriptors;

        store.release();

        // Add bw and gs images.
        imagemap[ID].pixSum = imread(gsdir + "/" + fn, CV_LOAD_IMAGE_GRAYSCALE);
        imagemap[ID].bw = imread(bwdir + "/" + fn, CV_LOAD_IMAGE_GRAYSCALE);
        
        // Check for invalid input
        if(! imagemap[ID].pixSum.data )
        {
            cout <<  "\033[1;31mCould not open or find pixsum for " << gsdir + "/" << fn << ".\033[0m" << endl; //]]
            return -1;
        }
        if(! imagemap[ID].pixSum.data ) // Check for invalid input
        {
            cout <<  "\033[1;31mCould not open or find bw for " << bwdir + "/" << fn << ".\033[0m" << endl; //]]
            return -1;
        }
    }

    cout << ">\n<\n  Analyzing Images ..." << endl;

    // Analyze the image to match against
    Mat image = imread(image_name);
    similarities::iandf matchingImage;
    matchingImage.im = image;

    if(! image.data ) // Check for invalid input
    {
        cout <<  "\033[1;31m  Could not open or find the image\033[0m" << endl; //]]
        return -1;
    }

    // Check for dimensions of loaded gs and bw images. This assumes they are all the same.
    map<vector<float>, similarities::iandf>::iterator item = imagemap.begin();
    int r = item->second.bw.rows;
    if (divs != r)
    {
        cout <<  "\033[1;33m  Requested " << divs << " divs but saved images have " << r << " divs.\033[0m" << endl; //]]
        divs = r;
    }

    matchingImage.pixSum = averageImage::getPixSumFromImage(image, divs);
    matchingImage.bw = averageImage::aboveBelow(matchingImage.pixSum);

    // Compute descriptors
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

    int done = 0;
    int total = imagemap.size();

    // Calculate simularities between image and all in database
    for (map<vector<float>, similarities::iandf>::iterator i = imagemap.begin(); i != imagemap.end(); ++i)
    {
        vector<float> ID = i->first;
        imagemap[ID].sim = similarities::compareIandFs(imagemap[ID], matchingImage);

        done++;
        if (done % 100 == 0)
        {
            cout << "  " << 100 * done / total << " percent done." << endl;
        }
    }

    map<vector<float>, int> top;
    bool gotone = false;

    typedef map<vector<float>, int>::iterator iter;

    cout << ">\n<\n  Choosing the Best Images ..." << endl;

    // Choose the best ones!
    for (map<vector<float>, similarities::iandf>::iterator i = imagemap.begin(); i != imagemap.end(); ++i)
    {
        vector<float> ID = i->first;
        int sim = imagemap[ID].sim;

        if (!gotone) // Have we already got one?
        {
            top[ID] = sim;
            gotone = true;
        }

        // Get the highest value currently in our top
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

        // Add it to top, replace an item in top, or move on
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

    namedWindow("Image");
    namedWindow("Match");
    namedWindow("ImageBW");
    namedWindow("MatchBW");
    namedWindow("ImageGS");
    namedWindow("MatchGS");

    imshow("Image", matchingImage.im);
    imshow("ImageBW", matchingImage.bw);
    imshow("ImageGS", matchingImage.pixSum);

    int count = 1;

    for (iter i = top.begin(); i != top.end(); ++i)
    {
        vector<float> ID = i->first;

        cout << "  Score: "<< i->second << endl;
        string fn = "Sim_" + boost::to_string(i->second) + "_" + boost::to_string(count) + ".png";
        imwrite(fn, imagemap[ID].im);

        count++;

        imshow("MatchBW", imagemap[ID].bw);
        imshow("Match", imagemap[ID].im);
        imshow("MatchGS", imagemap[ID].pixSum);

        waitKey(10000);
    }

    cout << ">\nComparisons took " << s << " seconds for " << imagemap.size() << " images (" 
        << (int) imagemap.size()/s << " images per second)." << endl;

    return 0;
}