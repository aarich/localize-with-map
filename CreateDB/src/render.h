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

    vector<float> rotateVector(vector<float> oldvector, float x, float y, float z)
{
    vector<float> newvector;

    // cout << "xyz: " <<  x << " " << y << " " << z << endl;

    // Roate about x
    newvector.push_back(oldvector[0]);
    newvector.push_back(oldvector[1] * cos(x) - oldvector[2] * sin(x));
    newvector.push_back(oldvector[1] * sin(x) + oldvector[2] * cos(x));

    // Rotate about y
    float tempx = newvector[0] * cos(y) + newvector[2] * sin(y);
    float tempy = newvector[1];
    float tempz = newvector[2] * cos(y) - newvector[0] * sin(y);

    newvector[0] = tempx;
    newvector[1] = tempy;
    newvector[2] = tempz;

    // Rotate about z
    tempx = newvector[0] * cos(z) - newvector[1] * sin(z);
    tempy = newvector[0] * sin(z) + newvector[1] * cos(z);
    tempz = newvector[2];

    newvector[0] = tempx;
    newvector[1] = tempy;
    newvector[2] = tempz;
    // cout << oldvector[0] << oldvector[1] << oldvector[2] << " -> " << newvector[0] << newvector[1] << newvector[2] << endl;

    return newvector;
}

vector<float> makeID(vector<float> pos, vector<float> dir)
{
    vector<float> newvector;

    newvector.push_back(pos[0]);
    newvector.push_back(pos[1]);
    newvector.push_back(pos[2]);
    newvector.push_back(dir[0]);
    newvector.push_back(dir[1]);
    newvector.push_back(dir[2]);

    return newvector;
}

map<vector<float>, Mat> createImages(PointCloud<PT>::Ptr cloud, int nop, int w, int h, float r, float f)
{
    cout << "<\n  Creating Images\n";

    map<vector<float>, Mat> images;

    // Find the bounding box so we know where to iterate through
    octree::OctreePointCloudSearch<PT> octree (r);
    octree.setInputCloud(cloud);
    octree.defineBoundingBox();
    double min_x, min_y, min_z;
    double max_x, max_y, max_z;
    octree.getBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z);

    // ====== Assuming z is the vertical direction!! ====== //

    // For now, we'll stay at the same z value
    double z = 0.0; //(max_z + min_z)/2;

    // What is the spacing between locations?
    double xdiv = (float)(max_x - min_x)/ (float)nop;
    double ydiv = (float)(max_y - min_y)/ (float)nop;

    // Initialize transformation object that will hold all travel transformations
    // move to the bottom corner of the map
    Eigen::Affine3f tf = Eigen::Affine3f::Identity();
    tf.translation() << min_x - xdiv/2, min_y - ydiv/2, z/2;
    transformPointCloud (*cloud, *cloud, tf);
    
    cout << "  Initializing Transforms ...\n";

    // Set up Yaw objects. We are going to turn around with 8 stops, then go back
    // Yaw per point
    int yawpp = 8;
    float yaw = 0.785398163;
    float yawback = -8 * 0.785398163;
    // Initialize yaw object
    Eigen::Affine3f yawtf = Eigen::Affine3f::Identity();
    yawtf.rotate (Eigen::AngleAxisf (yaw, Eigen::Vector3f::UnitZ()));
    // Initialize yawback object (To prevent numerical precision issues)
    Eigen::Affine3f yawbacktf = Eigen::Affine3f::Identity();
    yawbacktf.rotate (Eigen::AngleAxisf (yawback, Eigen::Vector3f::UnitZ()));

    // Pitch is for looking up and down.
    float pitchup = 0.523598776;
    float pitchdown = -1.047197552;
    // Initialize pitchup object
    Eigen::Affine3f pitchuptf = Eigen::Affine3f::Identity();
    pitchuptf.rotate (Eigen::AngleAxisf (pitchup, Eigen::Vector3f::UnitX()));
    // Initialize pitchdown object
    Eigen::Affine3f pitchdowntf = Eigen::Affine3f::Identity();
    pitchdowntf.rotate (Eigen::AngleAxisf (pitchdown, Eigen::Vector3f::UnitX()));


    // Initialize starting location and direction
    vector<float> location;
    location.push_back(min_x - xdiv/2); location.push_back(min_y - ydiv/2); location.push_back(z/2);

    vector<float> direction;
    direction.push_back(0); direction.push_back(1); direction.push_back(0);

    int todo = nop * nop * 3 * 8;

    cout << "  About to create " << todo << " images ...\n";

    int done = 0;

    for (int x = 0; x < nop; x++)
    {
        // Move to the new x!
        tf = Eigen::Affine3f::Identity();
        tf.translation() << xdiv, 0.0, 0.0;
        transformPointCloud(*cloud, *cloud, tf);

        location[0] += xdiv;

        for (int y = 0; y < nop; y++)
        {
            // cout << "  " << done << "/" << todo << " or " << 100 * done / todo << " percent done.\n";
            cout << 100 * done / todo << " percent done.\n";
            // Move to the new y!
            tf = Eigen::Affine3f::Identity();
            tf.translation() << 0.0, ydiv, 0.0;
            transformPointCloud(*cloud, *cloud, tf);

            location[1] += ydiv;

            // We're at a location in the map: (x, y, z) -- it looks like the origin.
            // First look up
            transformPointCloud(*cloud, *cloud, pitchuptf);
            direction = rotateVector(direction, pitchup, 0, 0);
            for (double i = 0; i < yawpp; i++)
            {
                transformPointCloud(*cloud, *cloud, yawtf);
                direction = rotateVector(direction, 0, 0, yaw);
                // cout << "  " << direction[0] << "\t" << direction[1] << "\t" << direction[2] << "\t" << endl;
                images[makeID(location, direction)] = render::makeImagefromSize(cloud, w, h, r, f);
            }
            transformPointCloud(*cloud, *cloud, yawbacktf);
            direction = rotateVector(direction, 0, 0, yawback);

            // Then down
            transformPointCloud(*cloud, *cloud, pitchdowntf);
            direction = rotateVector(direction, pitchdown, 0, 0);
            for (double i = 0; i < yawpp; i++)
            {
                transformPointCloud(*cloud, *cloud, yawtf);
                direction = rotateVector(direction, 0, 0, yaw);
                images[makeID(location, direction)] = render::makeImagefromSize(cloud, w, h, r, f);
            }
            transformPointCloud(*cloud, *cloud, yawbacktf);
            direction = rotateVector(direction, 0, 0, yawback);

            // Then back up to normal
            transformPointCloud(*cloud, *cloud, pitchuptf);
            direction = rotateVector(direction, pitchup, 0, 0);
            for (double i = 0; i < yawpp; i++)
            
{                transformPointCloud(*cloud, *cloud, yawtf);
                direction = rotateVector(direction, 0, 0, yaw);
                images[makeID(location, direction)] = render::makeImagefromSize(cloud, w, h, r, f);
            }
            transformPointCloud(*cloud, *cloud, yawbacktf);
            direction = rotateVector(direction, 0, 0, yawback);

            done += 3*8;
        }
        // We've gone through all y values. We should go back all the way to initial y.
        tf = Eigen::Affine3f::Identity();
        tf.translation() << 0.0, -1*nop*ydiv, 0.0;
        transformPointCloud(*cloud, *cloud, tf);
        location[1] -= nop * ydiv;
    }

    cout << "  Done.\n>" << endl;

    return images;
}

}