/*
#################################################################
## UTF-8
## �ļ�˵��������opencv�����ͼƬ��ȡ����С������չʾ��ͨ��MeanShift����д���
#################################################################
*/

#include <iostream>
#include "meanShift.h"


/*
#################################################################
## ������main
## ����������������ڡ��������ͼƬ�ļ�����ͼƬ��С���е�����չʾ���ս����
## ����������

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
