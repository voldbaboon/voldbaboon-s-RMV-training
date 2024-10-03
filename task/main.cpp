#include "windmill.hpp"
#include <ceres/ceres.h>
#include <iostream>
#include <vector>
#include <math.h>

using namespace std;
using namespace cv;

bool compareContourAreas(const vector<Point>& contour1, const vector<Point>& contour2) {
    return contourArea(contour1) < contourArea(contour2);
}

// 定义CostFunction结构体
struct CostFunction {
    CostFunction(const std::vector<double>& true_values) : true_values(true_values) {}

    template <typename T>
    bool operator()(const T* const A_get, const T* const w_get, const T* const fai_get, const T* const b_get, T* residual) const {
        // 计算速度并拟合正弦函数
        for (size_t i = 0; i < true_values.size(); ++i) {
            T velocity = T(true_values[i]); // 实际速度
            T predicted_velocity = A_get[0] * ceres::sin(w_get[0] * T(i) + fai_get[0]) + b_get[0];
            residual[i] = predicted_velocity - velocity;
        }

        return true;
    }

    std::vector<double> true_values;
};

//走过的距离
double distanceBetweenVectors(cv::Point2i v1, cv::Point2i v2) {
    double dot_product = v1.x - v2.x;
    double cross_product = v1.y - v2.y;
    
    return std::sqrt(dot_product * dot_product + cross_product * cross_product);
}

int main()
{
    std::chrono::milliseconds t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    WINDMILL::WindMill wm(t.count()/1000); //都采取秒做单位
    cv::Mat src;
    cv::Point2i R_Center, boxCenter;
    int count_frames = 0;         
    // 初始化拟合参数
    double A_get = 0.785 + 1, A = 0.785; // 初始值与真值相差至少1
    double w_get = 1.884 + 1, w = 1.884; // 初始值与真值相差至少1
    double fai_get = 1.65 + 1, fai = 1.65; // 初始值与真值相差至少1
    double b_get = 1.305 + 1, b = 1.305; 
    
    vector<Point2i> rPoints, boxPoints;
    vector<double> velocity;                          
    while (1)
    {
        t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        src = wm.getMat((double)t.count()/1000);
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
        if(count_frames > 0){
            double temp_v;//, r = cv::norm(R_Center - boxCenter);
            // 获取最后两个点的位置
            cv::Point2i R_Center1 = rPoints[count_frames - 1];
            cv::Point2i R_Center2 = rPoints[count_frames];
            cv::Point2i boxCenter1 = boxPoints[count_frames - 1];
            cv::Point2i boxCenter2 = boxPoints[count_frames];
            // 计算圆心到圆上点的向量
            cv::Point2i vec1 = boxCenter1 - R_Center1;
            cv::Point2i vec2 = boxCenter2 - R_Center2;
            cout << endl << "1" << vec1.x << "  " << vec1.y << endl;
            cout << endl << "2" << vec2.x << "  " << vec2.y << endl;
            
            // 计算两个向量的距离
            temp_v = abs(distanceBetweenVectors(vec1, vec2)); // 使用绝对值
            velocity.push_back(temp_v);
            cout << " haha" << temp_v;
            // 创建新的问题
            ceres::Problem problem;

            // 添加CostFunction到Problem中
            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunction, ceres::DYNAMIC, 1, 1, 1, 1>(new CostFunction(velocity), velocity.size());
            problem.AddResidualBlock(cost_function, nullptr, &A, &w, &fai, &b);
            // 设置优化选项
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.minimizer_progress_to_stdout = true;
            // 运行优化(每收集10个点)
            if (count_frames % 10 == 0 && count_frames > 0) {
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);}
            // 判断是否符合停止条件
            if (abs(0.785-A_get)<0.05*0.785 && abs(1.884-w_get)<0.05*1.884 && abs(1.65-fai_get)<0.05*1.65 &&abs(1.305-b_get)<0.05*1.305) {
                break;
            }
        }
        // 记录经过的帧数
        count_frames++;    
        cv::imshow("windmill", src);
        cv::waitKey(1);                  
        std::cout << "A_get: " << A_get << endl;
        std::cout << "w_get: " << w_get << endl;
        std::cout << "fai_get: " << fai_get << endl;
        std::cout << "b_get: " << b_get << endl;
        std::cout << "经过帧数：" << count_frames;
        //=======================================================//    
    }

}


