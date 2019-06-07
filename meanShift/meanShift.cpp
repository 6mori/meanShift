#include "meanShift.h"

using namespace std;

extern "C" void MSFiltering_d(Point3D *points, int width, int height, int hs, int hr, Point3D *temp_output);
extern "C" void MSSegmentation_d(Point3D *points, int width, int height, int hs, int hr, Point3D *temp_output);
extern "C" void setupPoints(Point3D *points, int width, int height, Point3D **points_d, Point3D **temp_output);
extern "C" void freePoints(Point3D *points, int width, int height, Point3D *points_d, Point3D *temp_output);

// Constructor
Point3D::Point3D() {
}

// Destructor
Point3D::~Point3D() {
}

// Scale the OpenCV Lab color to Lab range
void Point3D::PointLab() {
  l = l * 100 / 255;
  a = a - 128;
  b = b - 128;
}

// Sclae the Lab color to OpenCV range that can be used to transform to RGB
void Point3D::PointRGB() {
  l = l * 255 / 100;
  a = a + 128;
  b = b + 128;
}

// Accumulate points
void Point3D::MSPoint3DAccum(Point3D Pt) {
  l += Pt.l;
  a += Pt.a;
  b += Pt.b;
}

// Copy a point
void Point3D::MSPoint3DCopy(Point3D Pt) {
  l = Pt.l;
  a = Pt.a;
  b = Pt.b;
}

// Compute color space distance between two points
float Point3D::MSPoint3DColorDistance(Point3D Pt) {
  return sqrt((l - Pt.l) * (l - Pt.l) + (a - Pt.a) * (a - Pt.a) + (b - Pt.b) * (b - Pt.b));
}

/*
// Compute spatial space distance between two points
float Point3D::MSPoint3DSpatialDistance(Point3D Pt) {
  return sqrt((x - Pt.x) * (x - Pt.x) + (y - Pt.y) * (y - Pt.y));
}
*/

// Scale point
void Point3D::MSPoint3DScale(float scale) {
  l *= scale;
  a *= scale;
  b *= scale;
}

// Set point value
void Point3D::MSPoint3DSet(float pl, float pa, float pb) {
  l = pl;
  a = pa;
  b = pb;
}

// Print 5D point
void Point3D::Print() {
  cout << l << " " << a << " " << b << endl;
}

MeanShift::MeanShift(float s, float r) {
  hs = s;
  hr = r;
}

/*
#################################################################
## ������run
## ����������Ϊ�������������ڴ���Դ棬��RGB->Lab��ɫ�ռ�ת��������ʵ�ʲ���������
##         ��Lab->RGB��ɫ�ռ�ת������󽫲������д�أ��ͷ��ڴ���Դ档
## ����������
## Img������ͼ���cv::Mat�����д�ظ�cv::Mat
#################################################################
*/
void MeanShift::run(Mat & Img) {
  Point3D *points = new Point3D[Img.rows * Img.cols];
  Point3D *points_d = nullptr, *temp_output;
  for (int i = 0; i < Img.rows; i++) {
    for (int j = 0; j < Img.cols; j++) {
      Point3D point = Point3D();
      point.MSPoint3DSet(Img.at<Vec3b>(i, j)[0], Img.at<Vec3b>(i, j)[1], Img.at<Vec3b>(i, j)[2]);
      points[i * Img.cols + j] = point;
    }
  }
  setupPoints(points, Img.rows, Img.cols, &points_d, &temp_output);
  //MSFiltering(points_d, Img.rows, Img.cols, temp_output);
  MSSegmentation(points_d, Img.rows, Img.cols, temp_output);
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
## ������MSFiltering
## ����������
## ����������
## points������ԭͼ�����ص�λ�ú�Labֵ��Ϣ������Point3D�Դ�����
## width������ͼ��Ŀ�
## height������ͼ��ĸ�
## temp_output�����Point3D�Դ�����
#################################################################
*/
void MeanShift::MSFiltering(Point3D *points, int width, int height, Point3D *temp_output) {
  MSFiltering_d(points, width, height, hs, hr, temp_output);
}

/*
#################################################################
## ������MSSegmentation
## ����������
## ����������
## points������ԭͼ�����ص�λ�ú�Labֵ��Ϣ������Point3D�Դ�����
## width������ͼ��Ŀ�
## height������ͼ��ĸ�
## temp_output�����Point3D�Դ�����
#################################################################
*/
void MeanShift::MSSegmentation(Point3D *points, int width, int height, Point3D *temp_output) {
  MSSegmentation_d(points, width, height, hs, hr, temp_output);
}