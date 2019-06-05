#ifndef MEANSHIFT_H
#define MEANSHIFT_H

#include <vector>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"

using namespace cv;
using namespace std;

#define CHECK(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s \n", \
        error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

class Point3D {
public:
  float l;
  float a;
  float b;
public:
  Point3D();													// Constructor
  ~Point3D();													// Destructor
  void PointLab();											// Scale the OpenCV Lab color to Lab range
  void PointRGB();											// Sclae the Lab color to OpenCV range that can be used to transform to RGB
  void MSPoint3DAccum(Point3D);								// Accumulate points
  void MSPoint3DCopy(Point3D);								// Copy a point
  float MSPoint3DColorDistance(Point3D);						// Compute color space distance between two points
  //float MSPoint3DSpatialDistance(Point3D);					// Compute spatial space distance between two points
  void MSPoint3DScale(float);									// Scale point
  void MSPoint3DSet(float, float, float);		// Set point value
  void Print();
};

class MeanShift {
public:
  float hs;				// spatial radius
  float hr;				// color radius
public:
  MeanShift(float, float);		 							// Constructor for spatial bandwidth and color bandwidth
  void run(Mat&);
  void MSFiltering(Point3D *points, int width, int height, Point3D *temp_output);										// Mean Shift Filtering
  void MSSegmentation(Point3D *points, int width, int height);									// Mean Shift Segmentation
};

#endif