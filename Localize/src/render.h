#include <iostream>
#include <cstdlib>

#include <pcl/point_cloud.h>
#include <pcl/octree/octree.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <cvaux.h>

using namespace pcl;
using namespace cv;
using namespace std;

namespace render
{
    #define PT PointXYZRGB
    Mat makeImagefromSize(PointCloud<PT>::Ptr cloud, 
                const int width = 500, const int height = 500, float resolution = 0.02f, float f = 2.0f)
    {     
        octree::OctreePointCloudSearch<PT> octree (resolution);
        octree.setInputCloud(cloud);
        octree.defineBoundingBox();
        octree.addPointsFromInputCloud();
        
        Eigen::Vector3f   origin (0, 0.0f, 0.0f);
        Eigen::Vector3f location (0, 0.0f, 0.0f);
        vector<PT, Eigen::aligned_allocator<PT> > voxelCenterList;
        vector<int> k_indices;

        Mat result(width, height, CV_8UC3);
        result = Scalar(0, 0, 0);

        for (int w = 0; w < width; w++) 
        {
            for (int h = 0; h < height; h++)
            {
                voxelCenterList.clear();
                k_indices.clear();

                location(0) = -width/2 + w;
                location(1) = -height/2 + h;
                location(2) = -f;
                Eigen::Vector3f direction = -1 * location;

                // int nPoints = octree.getIntersectedVoxelCenters(origin, direction, voxelCenterList, 1);
                int nPoints = octree.getIntersectedVoxelIndices(origin, direction, k_indices, 1);

                if (nPoints > 0) 
                {
                    #if 0
                    cout << "Found " << nPoints << " points at "<< w << ", " << h << ".\n";
                    cout << "One Point: " << voxelCenterList[0] << endl;
                    Point3_<uchar>* p = result.ptr<Point3_<uchar> >(w, h);
                    p->x = (int) voxelCenterList[0].b;
                    p->y = (int) voxelCenterList[0].g;
                    p->z = (int) voxelCenterList[0].r;
                    #else
                    // cout << "Point: " << cloud->points[k_indices[0]] << endl;

                    PT point = cloud->points[k_indices[0]];
                    Point3_<uchar>* p = result.ptr<Point3_<uchar> >(h, w);
                    p->x = (int) point.b;
                    p->y = (int) point.g;
                    p->z = (int) point.r;
                    #endif
                }
            }
        }

        return result;
	}

}