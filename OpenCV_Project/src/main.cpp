#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
using namespace cv;
using namespace std;

class ImageDisplay {
public:
    void displayImages(const vector<Mat>& images, int numImagesPerRow, const string& name) {
        int numImages = images.size();
        if (numImages % numImagesPerRow != 0) {
            cout << "除不尽" << endl;
            return;
        }

        Mat combinedImage;
        vector<Mat> rows;

        // 计算目标尺寸，使用第一张图像的尺寸
        Size targetSize = images[0].size();

        for (int i = 0; i < numImages; i += numImagesPerRow) {
            vector<Mat> currentRow(images.begin() + i, images.begin() + i + numImagesPerRow);
            convertToThreeChannels(currentRow); // 确保所有图像都是3通道
            resizeImages(currentRow, targetSize); // 确保所有图像都缩放到相同尺寸
            Mat rowImage;
            hconcat(currentRow, rowImage);//把currentrow横向连起来，返回给rowImage
            rows.push_back(rowImage);
        }

        vconcat(rows, combinedImage);
        imshow(name, combinedImage);
        waitKey(0);
    }

private:
    void resizeImages(vector<Mat>& images, const Size& targetSize) {
        for (auto& image : images) { //范围基 for 循环（range-based for loop），用于遍历容器中的元素
            resize(image, image, targetSize); // 将每张图像缩放到目标尺寸
        }
    }

    void convertToThreeChannels(vector<Mat>& images) {
        for (auto& image : images) {
            if (image.channels() == 1) {
                cvtColor(image, image, COLOR_GRAY2BGR); // 将1通道图像转换为3通道
            }
        }
    }
};


int main()
{
//展示原始图片
Mat src = imread("/home/baboon/voldbaboon-s-RMV-training/OpenCV_Project/resources/test_image.png");
resize(src, src, Size(340,480));
imshow("initial_image", src);
/* 
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
    // 设定红色阈值范围
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

//绘制轮廓
    // 转换为灰度图像
    Mat gray_image;
    cvtColor(red_image, gray_image, COLOR_BGR2GRAY); // 转换为灰度图像
    // 二值化
    Mat binary;
    threshold(gray_image, binary, 10, 255, THRESH_BINARY); // 二值化
    // 查找轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE); //RETR_TREE是所有轮廓
    // 创建一个与原图同大小的图像用于绘制轮廓
    Mat contour_image = Mat::zeros(src.size(), CV_8UC3);  //CV_8UC3是一种图像数据类型，8U代表0～255,C3代表3个通道
    // 绘制轮廓
    for (size_t i = 0; i < contours.size(); i++) {//i用来遍历所有轮廓（包含在contours向量里）
        //int i 代表第几个， 2代表粗细
        drawContours(contour_image, contours, -1, Scalar(0, 255, 0), 2); // 绿色轮廓
    }
    // 显示带有轮廓的图像
    //imshow("Contours", contour_image);

//寻找红色区域的bounding box（包含这些红色区域的最小矩形）
    Mat boundingBox_image = Mat::zeros(src.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); i++) {
        Rect bounding_rect = boundingRect(contours[i]);
        rectangle(boundingBox_image, bounding_rect, Scalar(0, 255, 0), 2);
    }

    // 显示包含 bounding box 的图像
    //imshow("Green bounding boxes", boundingBox_image);

//计算轮廓面积（对于太短的孤立轮廓线则过滤掉）
    Mat size_image = Mat::zeros(src.size(), CV_8UC3);
        for (size_t i = 0; i < contours.size(); i++) {
            // 绘制轮廓
            drawContours(size_image, contours, (int)i, Scalar(0, 255, 0), 2);
            // 计算面积
            double area = contourArea(contours[i]);
            // 过滤掉孤立的线段
            if (arcLength(contours[i], true) > 10 && area > 10) {
                cout << "Contour " << i << " area: " << area << endl;
            } else {
                continue; //太小的就不显示
            }
        }

//提取高亮区域
Mat hsv_hl = hsv_image, hl_image;
Scalar hl_low = Scalar (0, 0, 200);
Scalar hl_high = Scalar (180, 255, 255);
Mat mask_hl;
inRange(hsv_hl, hl_low, hl_high, mask_hl);
bitwise_and(src, src, hl_image, mask_hl);
//imshow("hl_image", hl_image);

//高亮区域图形学处理
    //灰度化
    cvtColor(hl_image, hl_image, COLOR_BGR2GRAY);
    //imshow("hl_image", hl_image);
    //二值化
    threshold(hl_image, hl_image, 10, 255, THRESH_BINARY); // 二值化
    //膨胀
    // 定义腐蚀和膨胀的核（结构元素）
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(hl_image, hl_image, kernel); 
    // 腐蚀
	erode(hl_image, hl_image, kernel);
	//漫水
    cvtColor(hl_image, hl_image, COLOR_GRAY2BGR);
    Point seedpoint (339, 447);
    Scalar newcolor = (0, 0, 255);
    floodFill(hl_image, seedpoint, newcolor);

//图像绘制
    //绘制
    Mat draw_image = Mat::zeros(src.size(), CV_8UC3);
    // 绘制圆形
    cv::Point center(200, 200); // 圆心
    int radius = 50; // 半径
    cv::Scalar circleColor(0, 255, 0); // 绿色
    cv::circle(draw_image, center, radius, circleColor, -1); // -1 表示填充圆形

    // 绘制矩形
    cv::Point topLeft(50, 50); // 矩形的左上角
    cv::Point bottomRight(150, 150); // 矩形的右下角
    cv::Scalar rectangleColor(255, 0, 0); // 蓝色
    cv::rectangle(draw_image, topLeft, bottomRight, rectangleColor, -1); // -1 表示填充矩形

    // 绘制数字 "2"
    cv::Point textOrg(200, 300); // 文本的起始位置
    cv::Scalar textColor(0, 0, 255); 
    int fontFace = cv::FONT_HERSHEY_SIMPLEX; // 字体类型
    double fontScale = 2; // 字体大小
    int thickness = 2; // 字体线宽
    cv::putText(draw_image, "2", textOrg, fontFace, fontScale, textColor, thickness);

    //绘制红色外轮廓
    Mat drawGray_image;
    cvtColor(draw_image, drawGray_image, COLOR_BGR2GRAY); // 转换为灰度图像
    // 二值化
    Mat draw_binary;
    threshold(drawGray_image, draw_binary, 10, 255, THRESH_BINARY); // 二值化
    // 查找轮廓
    vector<vector<Point>> draw_contours;
    vector<Vec4i> draw_hierarchy;
    findContours(draw_binary, draw_contours, draw_hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    // 创建一个与原图同大小的图像用于绘制轮廓
    Mat drawContour_image = Mat::zeros(src.size(), CV_8UC3);  //CV_8UC3是一种图像数据类型，8U代表0～255,C3代表3个通道
    // 绘制轮廓
    for (size_t i = 0; i < draw_contours.size(); i++) {//i用来遍历所有轮廓（包含在contours向量里）
        drawContours(drawContour_image, draw_contours, -1, Scalar(0, 0, 255), 2); //红色轮廓
    }
    //imshow("drawContour_image", drawContour_image);

    //绘制红色bounding box
    Mat drawBoundingBox_image = Mat::zeros(src.size(), CV_8UC3);
    for (size_t i = 0; i < draw_contours.size(); i++) {
        Rect bounding_rect = boundingRect(draw_contours[i]);
        rectangle(drawBoundingBox_image, bounding_rect, Scalar(0, 0, 255), 2);
    }
    imshow("drawBoundingBox_image", drawBoundingBox_image);

//图像本身处理
    //旋转35度
    // 获取图像的中心点
    cv::Point2f rotated_center(src.cols / 2.0, src.rows / 2.0);
    // 计算旋转矩阵
    double angle = 35.0; // 旋转角度
    double scale = 1.0;  // 缩放因子
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(rotated_center, angle, scale);
    // 计算旋转后的图像大小
    cv::Rect2f bbox = cv::RotatedRect(rotated_center, src.size(), angle).boundingRect2f();
    // 调整旋转矩阵以考虑平移
    rotation_matrix.at<double>(0, 2) += bbox.width / 2.0 - center.x;
    rotation_matrix.at<double>(1, 2) += bbox.height / 2.0 - center.y;
    // 旋转图像
    cv::Mat rotated_image;
    cv::warpAffine(src, rotated_image, rotation_matrix, bbox.size());
    imshow("rotated_image", rotated_image);

    //裁剪为左上1/4
    int new_width = src.cols / 2;
    int new_height = src.rows / 2;
    // 定义区域（ROI）
    Rect roi(0, 0, new_width, new_height);

    // 裁剪图像
    Mat cropped_image = src(roi);

//展示图像
ImageDisplay imageDisplayer;
vector<Mat> images_task1 = {src, grey_image, hsv_image};      
vector<Mat> images_task2 = {MF_image, Gaussian_image};
vector<Mat> images_task3 = {red_image, contour_image, boundingBox_image, hl_image};
vector<Mat> images_task4 = {draw_image, drawContour_image, drawBoundingBox_image};
vector<Mat> images_task5 = {rotated_image, cropped_image};
imageDisplayer.displayImages(images_task1, 3, "图像颜色空间转换");
imageDisplayer.displayImages(images_task2, 2, "应用各种滤波操作");
imageDisplayer.displayImages(images_task3, 2, "特征提取");
imageDisplayer.displayImages(images_task4, 3, "图像绘制");
imageDisplayer.displayImages(images_task5, 2, "图像处理");

imageDisplayer.displayImages(images, 4, "combined window");
*/
 waitKey(0);
 return 0;
}