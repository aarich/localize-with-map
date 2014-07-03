#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "cvaux.h"

#include <pcl/io/vtk_lib_io.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/png_io.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common_headers.h>
#include <pcl/filters/filter.h>

#include "boost/filesystem.hpp"   

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>

#include "render.h"
#include "averageImage.h"
#include "Similarity.h"

using namespace cv;
using namespace std;
namespace fs = ::boost::filesystem;

// Print usage
static void printPrompt( const string& applName )
{
    cout << "/*\n"
    << " * Given a map and other parameters, this class will create a serialized map.\n"
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

    cv::initModule_nonfree();

    // Get Input Data
    ifstream file(argv[1]);
    if ( !file.is_open() )
        return false;
    
    string str;
    
        // Image Name
    getline( file, str ); getline( file, str );
    string image_name = str;
        // Cloud Name
    getline( file, str ); getline( file, str );
    string cloud_name = str;
        // width of images to be created.
    getline( file, str ); getline( file, str );
    int w = atoi(str.c_str());
        // height of images to be created
    getline( file, str ); getline( file, str );
    int h = atoi(str.c_str());
        // resolution of voxel grids
    getline( file, str ); getline( file, str );
    float r = atof(str.c_str());
        // f (distance from pinhole)
    getline( file, str ); getline( file, str );
    float f = atof(str.c_str());
        // thetax (initial rotation about X Axis of map)
    getline( file, str ); getline( file, str );
    float thetaX = atof(str.c_str());
        // thetay (initial rotation about Y Axis of map)
    getline( file, str ); getline( file, str );
    float thetaY = atof(str.c_str());
        // number of points to go to
    getline( file, str ); getline( file, str );
    float nop = atoi(str.c_str());
        // Number of divisions
    getline( file, str ); getline( file, str );
    float divs = atoi(str.c_str());
        // Number of images to return
    getline( file, str ); getline( file, str );
    int numtoreturn = atoi(str.c_str());    
        // Should we load or create photos?
    getline( file, str ); getline( file, str );
    string lorc =str.c_str();
        // Directory to look for photos
    getline( file, str ); getline( file, str );
    string dir =str.c_str();
        // save photos?
    getline( file, str ); getline( file, str );
    string savePhotos =str.c_str();
    
    file.close();
    // Done Getting Input Data

    map<vector<float>, Mat> imagemap;
    imagemap.clear();

    if ( !fs::exists( dir ) || lorc == "c" )
    { // Load Point Cloud and render images
        PointCloud<PT>::Ptr cloud (new pcl::PointCloud<PT>);
        io::loadPCDFile<PT>(cloud_name, *cloud);

        Eigen::Affine3f tf = Eigen::Affine3f::Identity();
        tf.rotate (Eigen::AngleAxisf (thetaX, Eigen::Vector3f::UnitX()));
        pcl::transformPointCloud (*cloud, *cloud, tf);
        tf = Eigen::Affine3f::Identity();
        tf.rotate (Eigen::AngleAxisf (thetaY, Eigen::Vector3f::UnitY()));
        pcl::transformPointCloud (*cloud, *cloud, tf);

        // Create images from point cloud
        imagemap = render::createImages(cloud, nop, w, h, r, f);

        if (savePhotos == "y")
        {
            for (map<vector<float>, Mat>::iterator i = imagemap.begin(); i != imagemap.end(); ++i)
            {
                string fn = dir + "/";
                for (int j = 0; j < i->first.size(); j++)
                    fn += boost::to_string(i->first[j]) + " ";
                fn += ".jpg";

                imwrite(fn, i->second);
            }
        }
    } 
    else 
    { // load images from the folder dir
        vector<fs::path> ret;
        const char * pstr = dir.c_str();
        fs::path p(pstr);
        get_all(pstr, ret);

        for (int i = 0; i < ret.size(); i++)
        {
            string fn = ret[i].string();
            istringstream iss(fn);
            vector<string> tokens;
            copy(istream_iterator<string>(iss),
               istream_iterator<string>(),
               back_inserter<vector<string> >(tokens));
            vector<float> ID;
            for (int i = 0; i < 6; i++) // 6 becuase there are three location floats and three direction floats
                ID.push_back(::atof(tokens[i].c_str()));
            fn = dir + "/" + fn;
            Mat m = imread(fn);

            imagemap[ID] = m;
        }
    }

    TickMeter tm;
    tm.reset();
    cout << "<\n  Analyzing Images ..." << endl;

    // We have a bunch of images, now we compute their grayscale and black and white.
    map<vector<float>, Mat> gsmap;
    map<vector<float>, Mat> bwmap;
    for (map<vector<float>, Mat>::iterator i = imagemap.begin(); i != imagemap.end(); ++i)
    {
        vector<float> ID = i->first;
        Mat Image = i-> second;
        GaussianBlur( Image, Image, Size(5,5), 0, 0, BORDER_DEFAULT );


        gsmap[ID] = averageImage::getPixSumFromImage(Image, divs);
        bwmap[ID] = averageImage::aboveBelow(gsmap[ID]);
    }
    Mat image = imread(image_name);
    Mat gsimage = averageImage::getPixSumFromImage(image, divs);
    Mat bwimage = averageImage::aboveBelow(gsimage);

    tm.start();


    cout << ">\n<\n  Comparing Images ..." << endl;

    // We have their features, now compare them!
    map<vector<float>, float> gssim; // Gray Scale Similarity
    map<vector<float>, float> bwsim; // Above Below Similarity

    for (map<vector<float>, Mat>::iterator i = gsmap.begin(); i != gsmap.end(); ++i)
    {
        vector<float> ID = i->first;
        gssim[ID] = similarities::getSimilarity(i->second, gsimage);
        bwsim[ID] = similarities::getSimilarity(bwmap[ID], bwimage); 
    }

    map<vector<float>, int> top;

    bool gotone = false;
    typedef map<vector<float>, int>::iterator iter;

    // Choose the best ones!
    for (map<vector<float>, Mat>::iterator i = imagemap.begin(); i != imagemap.end(); ++i)
    {
        vector<float> ID = i->first;

        int sim = gssim[ID] + 0.3*bwsim[ID];

        if (!gotone)
        {
            top[ID] = sim;
            gotone = true;
        }

        iter it = top.begin();
        iter end = top.end();
        int max_value = it->second;
        vector<float> max_ID = it->first;
        for( ; it != end; ++it) {
            int current = it->second;
            if(current > max_value) {
                max_value = it->second;
                max_ID = it->first;
            }
        }
        // cout << "Sim: " << sim << "\tmax_value: " << max_value << endl;
        if (top.size() <= numtoreturn)
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

    cout << ">\n<\n  Writing top " << numtoreturn << " images ..." << endl;

    for (iter i = top.begin(); i != top.end(); ++i)
    {
        vector<float> ID = i->first;
        for (int j = 0; j < 3; j++)
        {
            cout << ID[j] << "\t";
        }
        cout << "Similarity: "<< i->second << "\tGrayScale: " << gssim[ID] << "\tBW: " << bwsim[ID] << endl;
        string fn = "Sim_" + boost::to_string(i->second) + "_" + boost::to_string(gssim[ID]) + "_" + boost::to_string(bwsim[ID]) + ".png";

        imwrite(fn, imagemap[ID]);
    }

    tm.stop();
    double s = tm.getTimeSec();
    cout << ">\nComparisons took " << s << " seconds for " << imagemap.size() << " images (" 
        << (int) imagemap.size()/s << " images per second)." << endl;

    return 0;
}
