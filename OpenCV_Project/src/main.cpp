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
//imshow("grey_image", grey_image);

//HSV(H:色调 S：饱和度 V：明度)
Mat hsv_image;
cvtColor(src, hsv_image, COLOR_BGR2HSV);
//imshow("hsv_image", hsv_image);

//均值滤波
Mat MF_image;
blur(src, MF_image, Size(5,5)); 
//imshow("MF_image", MF_image);

//高斯滤波
Mat Gaussian_image;
GaussianBlur(src, Gaussian_image, Size(7,7), 2, 2);
//imshow("Gaussian_image", Gaussian_image);

//HSV方法提取红色区域
// 转换颜色空间为HSV（已做）
    Mat hsv1 = hsv_image, hsv2 = hsv_image;
    // 设定红色阈值范围（可以根据实际需求进行调整）
    Scalar lower_red1 = Scalar(0, 100, 100);
    Scalar upper_red1 = Scalar(10, 255, 255);

    Scalar lower_red2 = Scalar(156, 100, 100);
    Scalar upper_red2 = Scalar(180, 255, 255);
 
    // 通过阈值范围找到红色像素
    Mat mask1;
    inRange(hsv1, lower_red1, upper_red1, mask1);
    Mat mask2;
    inRange(hsv2, lower_red2, upper_red2, mask2);
 
    // 通过掩膜提取红色像素
    Mat red_image1, red_image2, red_image;
    bitwise_and(src, src, red_image1, mask1);
    bitwise_and(src, src, red_image2, mask2);
    
    // 显示提取后的红色像素图像
    red_image = red_image1 + red_image2;
    cv::imshow("Red Image", red_image);

 waitKey(0);
 return 0;
}