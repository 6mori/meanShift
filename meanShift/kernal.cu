/*
#################################################################
## UTF-8
## �ļ�˵�����ļ�����������cuda��صĺ�����
#################################################################
*/

#include "meanShift.h"

#include <stdio.h>

/*
#################################################################
## ������RGB2Lab
## ������������Point5D�����д�RGB�ռ�ת����Lab�ռ䡣
## ����������
## points������ԭͼ�����ص�λ�ú�RGBֵ��Ϣ������Point5D����
## width������ͼ��Ŀ�
## height������ͼ��ĸ�
#################################################################
*/
__global__ void RGB2Lab(Point5D *points, int width, int height) {
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  int tidy = blockDim.y * blockIdx.y + threadIdx.y;

  if (tidx < width && tidy < height) {
    float R = points[tidx + tidy * width].b / 255.0f;
    float G = points[tidx + tidy * width].a / 255.0f;
    float B = points[tidx + tidy * width].l / 255.0f;

    float XYZ[3];

    XYZ[0] = 0.412453f * R + 0.357580f * G + 0.180423f * B;
    XYZ[1] = 0.212671f * R + 0.715160f * G + 0.072169f * B;
    XYZ[2] = 0.019334f * R + 0.119193f * G + 0.950227f * B;

    XYZ[0] = XYZ[0] / 0.950456f;
    XYZ[2] = XYZ[2] / 1.088754f;

    float f[3];

    for (int i = 0; i < 3; i++) {
      if (XYZ[i] > 0.008856f) {
        f[i] = pow(XYZ[i], 1.0f / 3.0f);
      }
      else {
        f[i] = 7.787f * XYZ[i] + 0.137931f;
      }
    }

    if (XYZ[1] > 0.008856f) {
      points[tidx + tidy * width].l = 116.0f * pow(XYZ[1], 1.0f / 3.0f) - 16.0f;
    }
    else {
      points[tidx + tidy * width].l = 903.3f * XYZ[1];
    }
    points[tidx + tidy * width].a = 500.0f * (f[0] - f[1]);
    points[tidx + tidy * width].b = 200.0f * (f[1] - f[2]);
  }
}

/*
#################################################################
## ������RGB2Lab
## ������������Point5D�����д�Lab�ռ�ת����RGB�ռ䡣
## ����������
## points������ԭͼ�����ص�λ�ú�Labֵ��Ϣ������Point5D����
## width������ͼ��Ŀ�
## height������ͼ��ĸ�
#################################################################
*/
__global__ void Lab2RGB(Point5D *points, int width, int height) {
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  int tidy = blockDim.y * blockIdx.y + threadIdx.y;

  if (tidx < width && tidy < height) {
    float l = points[tidx + tidy * width].l;
    float a = points[tidx + tidy * width].a;
    float b = points[tidx + tidy * width].b;

    float f[3];

    f[1] = (l + 16.0f) / 116.0f;
    f[0] = a / 500.0f + f[1];
    f[2] = f[1] - b / 200.0f;

    for (int i = 0; i < 3; i++) {
      if (pow(f[i], 3) > 0.008856f) {
        f[i] = pow(f[i], 3);
      }
      else {
        f[i] = (f[i] - 0.137931f) / 7.787f;
      }
    }

    float X, Y, Z;

    X = f[0] * 0.950456f;
    Y = f[1];
    Z = f[2] * 1.088754f;

    float R = 3.240479f * X - 1.537150f * Y - 0.498535f * Z;
    float G = -0.969256f * X + 1.875992f * Y + 0.041556f * Z;
    float B = 0.055648f * X - 0.204043f * Y + 1.057311f * Z;

    points[tidx + tidy * width].l = B * 255.0f;
    points[tidx + tidy * width].a = G * 255.0f;
    points[tidx + tidy * width].b = R * 255.0f;
  }
}

/*
#################################################################
## ������setupPoints
## �������������Դ���ΪPoint5D�������ʱ�������ռ䣬��Point5D������ڴ渴�Ƶ��Դ��У�
##         �����н�Point5D�����RGB�ռ�ת����Lab�ռ䡣
## ����������
## points������ԭͼ�����ص�λ�ú�RGBֵ��Ϣ������Point5D�ڴ�����
## width������ͼ��Ŀ�
## height������ͼ��ĸ�
## points_d������ԭͼ�����ص�λ�ú�Labֵ��Ϣ�����Point5D�Դ�����
## temp_output�����ڴ洢�м������Դ�����
#################################################################
*/
extern "C"
__host__ void setupPoints(Point5D *points, int width, int height, Point5D **points_d, Point5D **temp_output) {
  CHECK(cudaMalloc((void**)points_d, sizeof(Point5D) * width * height));
  CHECK(cudaMalloc((void**)temp_output, sizeof(Point5D) * width * height));
  CHECK(cudaMemcpy(*points_d, points, sizeof(Point5D) * width * height, cudaMemcpyHostToDevice));
  dim3 threadsPerBlock(32, 32);
  dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
  cudaError_t error_check; 
  RGB2Lab<<<threadsPerBlock, blocksPerGrid >>>(*points_d, width, height);
  cudaDeviceSynchronize();
  error_check = cudaGetLastError();
  if (error_check != cudaSuccess) {
    printf("%s\n", cudaGetErrorString(error_check));
    system("pause");
  }
}

/*
#################################################################
## ������freePoints
## �������������н�Point5D�����RGB�ռ�ת����Lab�ռ䣬��������Դ渴�Ƶ��ڴ��У�
##         ���ͷ���������Դ档
## ����������
## points������ԭͼ�����ص�λ�ú�RGBֵ��Ϣ�����Point5D�ڴ�����
## width������ͼ��Ŀ�
## height������ͼ��ĸ�
## points_d������ԭͼ�����ص�λ�ú�Labֵ��Ϣ������Point5D�Դ�����
## temp_output�����ڴ洢�м������Դ�����
#################################################################
*/
extern "C"
__host__ void freePoints(Point5D *points, int width, int height, Point5D *points_d, Point5D *temp_output) {
  dim3 threadsPerBlock(32, 32);
  dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
  cudaError_t error_check;
  Lab2RGB<<<threadsPerBlock, blocksPerGrid>>>(points_d, width, height);
  cudaDeviceSynchronize();
  error_check = cudaGetLastError();
  if (error_check != cudaSuccess) {
    printf("%s\n", cudaGetErrorString(error_check));
    system("pause");
  }
  CHECK(cudaMemcpy(points, points_d, sizeof(Point5D) * width * height, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(temp_output));
  CHECK(cudaFree(points_d));
}

/*
#################################################################
## ������MSFiltering_d
## ����������
## ����������
## points������ԭͼ�����ص�λ�ú�Labֵ��Ϣ������Point5D�Դ�����
## width������ͼ��Ŀ�
## height������ͼ��ĸ�
## hs������ռ�뾶
## hr��������ɫ�뾶
## output�����Point5D�Դ�����
#################################################################
*/
extern "C" 
__host__ void MSFiltering_d(Point5D *points, int width, int height, int hs, int hr, Point5D *output) {
  printf("Hello from MSFiltering_d.\n");
}

/*
#################################################################
## ������MSSegmentation_d
## ����������
## ����������
## points������ԭͼ�����ص�λ�ú�Labֵ��Ϣ������Point5D�Դ�����
## width������ͼ��Ŀ�
## height������ͼ��ĸ�
## hs������ռ�뾶
## hr��������ɫ�뾶
## output�����Point5D�Դ�����
#################################################################
*/
extern "C" 
__host__ void MSSegmentation_d(Point5D *points, int width, int height, int hs, int hr) {
  printf("Hello from MSSegmentation_d.\n");
}

