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

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "render.h"
#include "averageImage.h"

using namespace cv;
using namespace std;

// Print usage
static void printPrompt( const string& applName )
{
    cout << "/*\n"
         << " * Given a map and other parameters, this class will create a serialized map.\n"
         << " */\n" << endl;

    cout << endl << "Format:\n" << endl;
    cout << "./" << applName << " [Inputs.txt]" << endl;
    cout << endl;
}

int main(int argc, char** argv)
{
    if( argc < 2 )
    {
        printPrompt( argv[0] );
        return -1;
    }

    cv::initModule_nonfree();

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
    file.close();

    PointCloud<PT>::Ptr cloud (new pcl::PointCloud<PT>);
    io::loadPCDFile<PT>(cloud_name, *cloud);

    Eigen::Affine3f tf = Eigen::Affine3f::Identity();
    tf.rotate (Eigen::AngleAxisf (thetaX, Eigen::Vector3f::UnitX()));
    pcl::transformPointCloud (*cloud, *cloud, tf);
    tf = Eigen::Affine3f::Identity();
    tf.rotate (Eigen::AngleAxisf (thetaY, Eigen::Vector3f::UnitY()));
    pcl::transformPointCloud (*cloud, *cloud, tf);

    // Create images from a point cloud
    map<vector<float>, Mat> imagemap = render::createImages(cloud, nop, w, h, r, f);
    
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

    cout << ">\n<\n  Comparing Images ..." << endl;

    // We have their features, now compare them!
    map<vector<float>, float> gssim;
    map<vector<float>, float> bwsim;
    for (map<vector<float>, Mat>::iterator i = gsmap.begin(); i != gsmap.end(); ++i)
    {
        vector<float> ID = i->first;
        gssim[ID] = averageImage::determineSimilarity(i->second, gsimage);
        bwsim[ID] = averageImage::determineSimilarity(bwmap[ID], bwimage); 
    }

    // namedWindow("Images");

    map<vector<float>, int> top;

    bool gotone = false;
    typedef map<vector<float>, int>::iterator iter;

    // Choose the best ones!
    for (map<vector<float>, Mat>::iterator i = imagemap.begin(); i != imagemap.end(); ++i)
    {
        vector<float> ID = i->first;

        // imshow("Images", i->second);
        // waitKey(0);

        int GSSIM = gssim[ID];
        int BWSIM = bwsim[ID];

        int sim = GSSIM + 0.3*BWSIM;

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

    cout << ">\n<\n  Writing Images ..." << endl;

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
    cout << ">" << endl;

    return 0;
}
