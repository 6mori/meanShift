#include "meanShift.h"

using namespace std;

extern "C" void MSFiltering_d(Point5D *points, int width, int height, int hs, int hr, Point5D *temp_output);
extern "C" void MSSegmentation_d(Point5D *points, int width, int height, int hs, int hr);
extern "C" void setupPoints(Point5D *points, int width, int height, Point5D **points_d, Point5D **temp_output);
extern "C" void freePoints(Point5D *points, int width, int height, Point5D *points_d, Point5D *temp_output);

// Constructor
Point5D::Point5D() {
  x = -1;
  y = -1;
}

// Destructor
Point5D::~Point5D() {
}

// Scale the OpenCV Lab color to Lab range
void Point5D::PointLab() {
  l = l * 100 / 255;
  a = a - 128;
  b = b - 128;
}

// Sclae the Lab color to OpenCV range that can be used to transform to RGB
void Point5D::PointRGB() {
  l = l * 255 / 100;
  a = a + 128;
  b = b + 128;
}

// Accumulate points
void Point5D::MSPoint5DAccum(Point5D Pt) {
  x += Pt.x;
  y += Pt.y;
  l += Pt.l;
  a += Pt.a;
  b += Pt.b;
}

// Copy a point
void Point5D::MSPoint5DCopy(Point5D Pt) {
  x = Pt.x;
  y = Pt.y;
  l = Pt.l;
  a = Pt.a;
  b = Pt.b;
}

// Compute color space distance between two points
float Point5D::MSPoint5DColorDistance(Point5D Pt) {
  return sqrt((l - Pt.l) * (l - Pt.l) + (a - Pt.a) * (a - Pt.a) + (b - Pt.b) * (b - Pt.b));
}

// Compute spatial space distance between two points
float Point5D::MSPoint5DSpatialDistance(Point5D Pt) {
  return sqrt((x - Pt.x) * (x - Pt.x) + (y - Pt.y) * (y - Pt.y));
}

// Scale point
void Point5D::MSPoint5DScale(float scale) {
  x *= scale;
  y *= scale;
  l *= scale;
  a *= scale;
  b *= scale;
}

// Set point value
void Point5D::MSPOint5DSet(float px, float py, float pl, float pa, float pb) {
  x = px;
  y = py;
  l = pl;
  a = pa;
  b = pb;
}

// Print 5D point
void Point5D::Print() {
  cout << x << " " << y << " " << l << " " << a << " " << b << endl;
}

MeanShift::MeanShift(float s, float r) {
  hs = s;
  hr = r;
}

/*
#################################################################
## 函数：run
## 函数描述：为后续操作申请内存和显存，做RGB->Lab颜色空间转换，调用实际操作函数，
##         做Lab->RGB颜色空间转换，最后将操作结果写回，释放内存和显存。
## 参数描述：
## Img：输入图像的cv::Mat，结果写回该cv::Mat
#################################################################
*/
void MeanShift::run(Mat & Img) {
  Point5D *points = new Point5D[Img.rows * Img.cols];
  Point5D *points_d = nullptr, *temp_output;
  for (int i = 0; i < Img.rows; i++) {
    for (int j = 0; j < Img.cols; j++) {
      Point5D point = Point5D();
      point.MSPOint5DSet(i, j, Img.at<Vec3b>(i, j)[0], Img.at<Vec3b>(i, j)[1], Img.at<Vec3b>(i, j)[2]);
      points[i * Img.cols + j] = point;
    }
  }
  setupPoints(points, Img.rows, Img.cols, &points_d, &temp_output);
  MSFiltering(points_d, Img.rows, Img.cols, temp_output);
  MSSegmentation(points_d, Img.rows, Img.cols);
  freePoints(points, Img.rows, Img.cols, points_d, temp_output);
  for (int i = 0; i < Img.rows; i++) {
    for (int j = 0; j < Img.cols; j++) {
      Img.at<Vec3b>(i, j)[0] = points[i * Img.cols + j].l;
      Img.at<Vec3b>(i, j)[1] = points[i * Img.cols + j].a;
      Img.at<Vec3b>(i, j)[2] = points[i * Img.cols + j].b;
    }
  }
  delete [] points;
}

/*
#################################################################
## 函数：MSFiltering
## 函数描述：
## 参数描述：
## points：包含原图像像素点位置和Lab值信息的输入Point5D显存数组
## width：输入图像的宽
## height：输入图像的高
## temp_output：输出Point5D显存数组
#################################################################
*/
void MeanShift::MSFiltering(Point5D *points, int width, int height, Point5D *temp_output) {
  MSFiltering_d(points, width, height, hs, hr, temp_output);
}

/*
#################################################################
## 函数：MSSegmentation
## 函数描述：
## 参数描述：
## points：包含原图像像素点位置和Lab值信息的输入Point5D显存数组
## width：输入图像的宽
## height：输入图像的高
## temp_output：输出Point5D显存数组
#################################################################
*/
void MeanShift::MSSegmentation(Point5D *points, int width, int height) {
  MSSegmentation_d(points, width, height, hs, hr);
}