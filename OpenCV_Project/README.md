# 第二次任务完成

## 图像颜色空间转换
- 灰度图和HSV都是使用cvtColor,第一个参数传入，第二个传出，第三个为使用的方法

## 应用各种滤波操作
- 需用Size指定滤波核的大小
- 均值滤波：blur； 高斯滤波：GaussianBlur， 最后需给出X、Y方向的高斯核的标准差（我都使用的是2）

## 特征提取
- 需要注意HSV空间中红色有两个阈值范围
```CPP
Scalar lower_red1 = Scalar(0, 100, 100);
    Scalar upper_red1 = Scalar(10, 255, 255);

    Scalar lower_red2 = Scalar(156, 100, 100);
    Scalar upper_red2 = Scalar(180, 255, 255);
```
