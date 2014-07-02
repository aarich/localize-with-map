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

#include "render.h"

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
    // theta (initial rotation about Y Axis of map)
    getline( file, str ); getline( file, str );
    float theta = atof(str.c_str());
    // theta (initial rotation about Y Axis of map)
    getline( file, str ); getline( file, str );
    float nop = atoi(str.c_str());

    file.close();

    PointCloud<PT>::Ptr cloud (new pcl::PointCloud<PT>);
    io::loadPCDFile<PT>(cloud_name, *cloud);

    Eigen::Affine3f tf = Eigen::Affine3f::Identity();
    tf.rotate (Eigen::AngleAxisf (theta, Eigen::Vector3f::UnitY()));
    pcl::transformPointCloud (*cloud, *cloud, tf);

    // Create images from a point cloud
    map<vector<float>, Mat> imagemap = render::createImages(cloud, nop, w, h, r, f);
    
    // We have a bunch of images, now we compute their grayscale and black and white.

    

    return 0;
}
