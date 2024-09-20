#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;


int main()
{
//展示原始图片
 Mat src = imread("/home/baboon/voldbaboon-s-RMV-training/OpenCV_Project/resources/test_image.png");
  imshow("initial_image", src);
  
//灰度图
 Mat grey_image;
 cvtColor(src,grey_image,COLOR_BGR2GRAY);
 imshow("grey_image", grey_image);

//HSV(H:色调 S：饱和度 V：明度)
 Mat hsv_image;
 cvtColor(src, hsv_image, COLOR_BGR2HSV);
 imshow("hsv_image", hsv_image);
 waitKey(0);
 return 0;
}