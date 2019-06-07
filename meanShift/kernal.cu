/*
#################################################################
## UTF-8
## 文件说明：文件包含所有与cuda相关的函数。
#################################################################
*/

#include "meanShift.h"

#include <stdio.h>

/*
#################################################################
## 函数：RGB2Lab
## 函数描述：将Point3D数组中从RGB空间转换到Lab空间。
## 参数描述：
## points：包含原图像像素点位置和RGB值信息的输入Point3D数组
## width：输入图像的宽
## height：输入图像的高
#################################################################
*/
__global__ void RGB2Lab(Point3D *points, int width, int height) {
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
## 函数：RGB2Lab
## 函数描述：将Point3D数组中从Lab空间转换到RGB空间。
## 参数描述：
## points：包含原图像像素点位置和Lab值信息的输入Point3D数组
## width：输入图像的宽
## height：输入图像的高
#################################################################
*/
__global__ void Lab2RGB(Point3D *points, int width, int height) {
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
## 函数：setupPoints
## 函数描述：在显存中为Point3D数组和临时输出申请空间，将Point3D数组从内存复制到显存中，
##         并并行将Point3D数组从RGB空间转换到Lab空间。
## 参数描述：
## points：包含原图像像素点位置和RGB值信息的输入Point3D内存数组
## width：输入图像的宽
## height：输入图像的高
## points_d：包含原图像像素点位置和Lab值信息的输出Point3D显存数组
## temp_output：用于存储中间结果的显存数组
#################################################################
*/
extern "C"
__host__ void setupPoints(Point3D *points, int width, int height, Point3D **points_d, Point3D **temp_output) {
  CHECK(cudaMalloc((void**)points_d, sizeof(Point3D) * width * height));
  CHECK(cudaMalloc((void**)temp_output, sizeof(Point3D) * width * height));
  CHECK(cudaMemcpy(*points_d, points, sizeof(Point3D) * width * height, cudaMemcpyHostToDevice));
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
## 函数：freePoints
## 函数描述：并行将Point3D数组从RGB空间转换到Lab空间，将结果从显存复制到内存中，
##         并释放申请过的显存。
## 参数描述：
## points：包含原图像像素点位置和RGB值信息的输出Point3D内存数组
## width：输入图像的宽
## height：输入图像的高
## points_d：包含原图像像素点位置和Lab值信息的输入Point3D显存数组
## temp_output：用于存储中间结果的显存数组
#################################################################
*/
extern "C"
__host__ void freePoints(Point3D *points, int width, int height, Point3D *points_d, Point3D *temp_output) {
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
  CHECK(cudaMemcpy(points, temp_output, sizeof(Point3D) * width * height, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(temp_output));
  CHECK(cudaFree(points_d));
}

/*
#################################################################
## 函数：MSFiltering_d
## 函数描述：
## 参数描述：
## points：包含原图像像素点位置和Lab值信息的输入Point3D显存数组
## width：输入图像的宽
## height：输入图像的高
## hs：输入空间半径
## hr：输入颜色半径
## output：输出Point3D显存数组
#################################################################
*/
extern "C" 
__host__ void MSFiltering_d(Point3D *points, int width, int height, int hs, int hr, Point3D *output) {
  printf("Hello from MSFiltering_d.\n");
}

/*
#################################################################
## 函数：MSSegmentation_d
## 函数描述：
## 参数描述：
## points：包含原图像像素点位置和Lab值信息的输入Point3D显存数组
## width：输入图像的宽
## height：输入图像的高
## hs：输入空间半径
## hr：输入颜色半径
## output：输出Point3D显存数组
#################################################################
*/
extern "C" 
__host__ void MSSegmentation_d(Point3D *points, int width, int height, int hs, int hr, Point3D *output) {
  printf("Hello from MSSegmentation_d.\n");
}

