/*
#################################################################
## UTF-8
## 文件说明：调用opencv库进行图片读取，大小调整和展示，通过MeanShift类进行处理。
#################################################################
*/

#include <iostream>
#include "meanShift.h"


/*
#################################################################
## 函数：main
## 函数描述：程序入口。负责读入图片文件并对图片大小进行调整，展示最终结果。
## 参数描述：

#################################################################
*/
int main() {
  Mat Img = imread("tsumugi.png");

  resize(Img, Img, Size(256, 256), 0, 0, 1);

  namedWindow("The Original Picture");
  imshow("The Original Picture", Img);

  MeanShift MS(8, 16);
  MS.run(Img);

  namedWindow("MS result");
  imshow("MS result", Img);

  waitKey(0);

  return 0;
}
