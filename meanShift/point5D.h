#include <iostream>
#include <stdio.h>
#include <cmath>
using namespace std ;


class Point5D{

public:

    float x;			// Spatial value

    float y;			// Spatial value

    float l;			// Lab value

    float a;			// Lab value

    float b;			// Lab value

public:

    __device__ Point5D(){ x = -1 ; y = - 1 ; } ;

    __device__ void PointLab();											// Scale the OpenCV Lab color to Lab range

    __device__ void PointRGB();											// Sclae the Lab color to OpenCV range that can be used to transform to RGB

    __device__  void MSPoint5DAccum(Point5D);								// Accumulate points

    __device__ void MSPoint5DCopy(Point5D);								// Copy a point

    __device__ float MSPoint5DColorDistance(Point5D);						// Compute color space distance between two points

    __device__ float MSPoint5DSpatialDistance(Point5D);					// Compute spatial space distance between two points

    __device__ void MSPoint5DScale(float);									// Scale point

    __device__ void MSPOint5DSet(float, float, float, float, float);		// Set point value

    __device__ void Print();												// Print 5D point

};






// Scale the OpenCV Lab color to Lab range

void Point5D::PointLab(){

	l = l * 100 / 255;

	a = a - 128;

	b = b - 128;

} ;



// Sclae the Lab color to OpenCV range that can be used to transform to RGB

void Point5D::PointRGB(){

	l = l * 255 / 100;

	a = a + 128;

	b = b + 128;

} ;



// Accumulate points

void Point5D::MSPoint5DAccum(Point5D Pt){

	x += Pt.x;

	y += Pt.y;

	l += Pt.l;

	a += Pt.a;

	b += Pt.b;

} ;



// Copy a point

void Point5D::MSPoint5DCopy(Point5D Pt){

	x = Pt.x;

	y = Pt.y;

	l = Pt.l;

	a = Pt.a;

	b = Pt.b;

} ;



// Compute color space distance between two points

float Point5D::MSPoint5DColorDistance(Point5D Pt){

	return sqrt((l - Pt.l) * (l - Pt.l) + (a - Pt.a) * (a - Pt.a) + (b - Pt.b) * (b - Pt.b));

} ;



// Compute spatial space distance between two points

float Point5D::MSPoint5DSpatialDistance(Point5D Pt){

	return sqrt((x - Pt.x) * (x - Pt.x) + (y - Pt.y) * (y - Pt.y));

} ;



// Scale point

void Point5D::MSPoint5DScale(float scale){

	x *= scale;

	y *= scale;

	l *= scale;

	a *= scale;

	b *= scale;

} ;



// Set point value

void Point5D::MSPOint5DSet(float px, float py, float pl, float pa, float pb){

	x = px;

	y = py;

	l = pl;

	a = pa;

	b = pb;

} ;



// Print 5D point

void Point5D::Print(){

	printf( " %f %f %f %f %f \n" , x , y , l , a , b ) ;

} ;
