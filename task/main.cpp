#include "windmill.hpp"
#include <ceres/ceres.h>
#include <iostream>
#include <vector>
#include <math.h>

using namespace std;
using namespace cv;

int N = 1600;
bool compareContourAreas(const vector<Point>& contour1, const vector<Point>& contour2) {
    return contourArea(contour1) < contourArea(contour2);
}

// 定义CostFunction结构体
struct CostFunction {
    CostFunction(double true_values, double dt) : true_values(true_values), dt(dt) {}
    template <typename T>
    bool operator()(const T* const num, T* residual) const {
        // 计算速度并拟合正弦函数
        T myCos = T(true_values); // 实际角度
        T myTime = T(dt);
        T predicted_myCos = cos(num[3]*myTime + (num[0]/num[1])*
            (ceres::cos(num[1]*T(0)+num[2]+3.14150265/2) - ceres::cos(num[1]*(T(0)+myTime)+num[2]+3.14159265/2)));
        predicted_myCos = predicted_myCos / T(3.1415926) * T(180);
        residual[0] = abs(predicted_myCos - myCos);
        return true;
    }
    double true_values;
    double dt;
};

int main()
{
    std::chrono::milliseconds t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    double t_start = (double)t.count();
    WINDMILL::WindMill wm(t_start); 
    cv::Mat src;
    cv::Point2i R_Center, boxCenter;
    int count_frames = 0;         
    // 初始化拟合参数
    double A_get = 0.785 + 1, A = 0.785; // 初始值与真值相差至少1
    double w_get = 1.884 + 1, w = 1.884;
    double fai_get = 0.24 + 1, fai = 0.24; 
    double b_get = 1.305 + 1, b = 1.305; 
    double num[4] = {1.785, 2.884, 1.24, 2.305};
    vector<Point2i> rPoints, boxPoints;
    double myCos[10000], dt[10000];                          
    while (count_frames < N)
    {
        t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        double t_now = (double)t.count() / 1000;
        src = wm.getMat(t_now);
        //==========================代码区========================//
        //识别程序
        cv::Mat grey_image, binary;
        cv::cvtColor(src, grey_image, COLOR_BGR2GRAY);
        threshold(grey_image, binary, 10 ,255, THRESH_BINARY);
        vector<vector<Point>> contours;
        vector<Point> R_contour, Hammer_contour;
        vector<Vec4i> hierarchy;
        findContours(binary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);    
        std::sort(contours.begin(), contours.end(), compareContourAreas);//按面积从小到大排列，最小的是R，其次是锤子
        //画R的中心
        if (contours[0].size() > 0){
            Rect bounding_rect = boundingRect(contours[0]);
            R_Center = cv::Point2i(bounding_rect.x + bounding_rect.width / 2, bounding_rect.y + bounding_rect.height / 2);
            cv::circle(src, R_Center, 8, Scalar(0, 255, 0), 1);
        }
        //画锤子的中心
        if (contours[1].size() > 0) {
            Rect bounding_rect = boundingRect(contours[1]);
            cv::Mat tempFind = binary(bounding_rect);
            vector<vector<Point>> tempFindContours;
            vector<Vec4i> tempFindHierarchy;
            findContours(tempFind, tempFindContours, tempFindHierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
            for (int j = 0; j < tempFindContours.size(); j++) {
                if (tempFindHierarchy[j][3] != -1) { // 内部轮廓
                    Rect bounding_findbox = boundingRect(tempFindContours[j]);
                    bounding_findbox.x += bounding_rect.x;
                    bounding_findbox.y += bounding_rect.y;
                    boxCenter = cv::Point2i(bounding_findbox.x + bounding_findbox.width / 2, bounding_findbox.y + bounding_findbox.height / 2);
                    cv::circle(src, boxCenter, 8, Scalar(0, 255, 0), 1);
                }
            }
        }

        rPoints.push_back(R_Center);
        boxPoints.push_back(boxCenter);
        double r = cv::norm(R_Center - boxCenter);
        // 获取最后点的位置
        cv::Point2i R_Center = rPoints[count_frames];
        cv::Point2i boxCenter = boxPoints[count_frames];
        // 计算圆心到圆上点的向量
        cv::Point2i vec = boxCenter - R_Center;
        // 计算角度
        dt[count_frames] = t_now - t_start;
        myCos[count_frames] = vec.x / r;
        count_frames++;// 记录经过的帧数       
        cv::imshow("windmill", src);
        cv::waitKey(1);     
    }
    // 创建新的问题
    ceres::Problem problem;
    for(size_t i = 0 ; i < count_frames; i++){
        // 添加CostFunction到Problem中
        double nowCos = myCos[count_frames - i];
        double nowDt = dt[count_frames-i];
        ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunction, 1, 4>(new CostFunction(nowCos, nowDt));
        problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(0.5), num);
        // 设置优化选项
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
    }
    // 判断条件
    if (abs(0.785-A_get)<0.05*0.785 && abs(1.884-w_get)<0.05*1.884 && abs(0.24-fai_get)<0.05*0.24 &&abs(1.305-b_get)<0.05*1.305 && count_frames > 10) {
    std::cout << endl << "finally" << endl;
    std::cout << "A_get: " << A_get << endl;
    std::cout << "w_get: " << w_get << endl;
    std::cout << "fai_get: " << fai_get << endl;
    std::cout << "b_get: " << b_get << endl;
    std::cout << "经过帧数：" << count_frames;
    }
    else
    {
        cout << endl << "fail" << endl;
    }
        //=======================================================//    
}