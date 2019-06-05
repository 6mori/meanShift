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

class Point5D {
public:
  float x;
  float y;
  float l;
  float a;
  float b;
public:
  Point5D();													// Constructor
  ~Point5D();													// Destructor
  void PointLab();											// Scale the OpenCV Lab color to Lab range
  void PointRGB();											// Sclae the Lab color to OpenCV range that can be used to transform to RGB
  void MSPoint5DAccum(Point5D);								// Accumulate points
  void MSPoint5DCopy(Point5D);								// Copy a point
  float MSPoint5DColorDistance(Point5D);						// Compute color space distance between two points
  float MSPoint5DSpatialDistance(Point5D);					// Compute spatial space distance between two points
  void MSPoint5DScale(float);									// Scale point
  void MSPOint5DSet(float, float, float, float, float);		// Set point value
  void Print();
};

class MeanShift {
public:
  float hs;				// spatial radius
  float hr;				// color radius
public:
  MeanShift(float, float);		 							// Constructor for spatial bandwidth and color bandwidth
  void run(Mat&);
  void MSFiltering(Point5D *points, int width, int height, Point5D *temp_output);										// Mean Shift Filtering
  void MSSegmentation(Point5D *points, int width, int height);									// Mean Shift Segmentation
};

#endif