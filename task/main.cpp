#include "windmill.hpp"
#include <ceres/ceres.h>
#include <iostream>
#include <vector>
#include <math.h>

using namespace std;
using namespace cv;

int N = 3000;//3000个取样点
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
        T predicted_myCos = cos(num[3]*myTime + (num[0]/num[1])*(ceres::cos(num[2]) - ceres::cos(num[1]*myTime+num[2])));
        residual[0] = myCos - predicted_myCos;
        return true;
    }
    double true_values;
    double dt;
};

int main()
{   double result[5] = {0};
    for(int test =  0; test < 10; test++){
        std::chrono::milliseconds t_s = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        double t_start = (double)t_s.count();
        WINDMILL::WindMill wm(t_start/1000); 
        cv::Mat src;
        cv::Point2i R_Center, boxCenter;
        int count_frames = 0;         
        // 初始化拟合参数
        double num[4] = {0.785+0.7, 1.884+0.7, 1.81+0.7, 1.305+0.7};
        double myCos[10000], dt[10000];                          
        while (count_frames < N)
        {
            std::chrono::milliseconds t_n = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
            double t_now = (double)t_n.count();
            src = wm.getMat(t_now / 1000);
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
            // 计算圆心到圆上点的向量
            cv::Point2i vec =  boxCenter - R_Center;
            double r = sqrt(vec.x*vec.x + vec.y*vec.y);
            dt[count_frames] = (t_now - t_start) / 1000;
            //cout << dt[count_frames%10] << endl;
            myCos[count_frames] = vec.x / r;
            count_frames++;// 记录经过的帧数       
            cv::imshow("windmill", src);
            cv::waitKey(1);     
        }
        // 创建新的问题
        ceres::Problem problem;
        for(size_t i = 0 ; i < N; i++){
            // 添加CostFunction到Problem中
            double nowCos = myCos[i];
            double nowDt = dt[i];
            //ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunction, 1, 4>(new CostFunction(nowCos, nowDt));
            problem.AddResidualBlock(new ceres::AutoDiffCostFunction<CostFunction, 1, 4>(new CostFunction(nowCos, nowDt))
                , new ceres::CauchyLoss(1.0), num);
        }
        // 设置优化选项
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;

        //options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        ceres::Solve(options, &problem, &summary);
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();//记录优化完的时间
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
        if(num[0] < 0) num[0] = -num[0];
        if(num[1] < 0) num[1] = -num[1];
        while(num[2] < 0) {num[2] += 6.2831;}
        while(num[2] > 6.2831) {num[2] -= 6.2831;}
        std::cout << endl <<"总用时: " << dt[count_frames-1]+(double)time_used.count() << endl;        
        for(int k = 0; k < 4; k++){result[k] += num[k];}
        result[4] += dt[count_frames-1]+(double)time_used.count();
        // 判断条件
        if (abs(0.785-num[0])<0.05*0.785 && abs(1.884-num[1])<0.05*1.884 && abs(1.81-num[2])<0.05*1.81 &&abs(1.305-num[3])<0.05*1.305) {
            std::cout << endl << "finally" << endl;
            std::cout << "A_get: " << num[0] << endl;
            std::cout << "w_get: " << num[1] << endl;
            std::cout << "fai_get: " << num[2] << endl;
            std::cout << "b_get: " << num[3] << endl;
            }   
        //=======================================================//    
    }
    for(int k = 0; k < 5; k++){
        std::cout << endl << result[k]/10 << endl;
    }
}
