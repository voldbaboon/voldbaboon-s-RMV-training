# 第二次任务完成

## 展示图片
- 本次任务要展示的图片很多，一个一个imshow不太方便，故写了一个用于展示图片的类，只包含三个函数，调整图片大小使其一致的
resizeImages，将1通道图片转变成3通道的convertToThreeChannels和最终展示的displayImages
- 注意拼接图片的函数hconcat和vconcat都要求图片大小、通道数一致，所以上面的两个函数是有必要的，另外因此我的图都是3通道，组长不要误会


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

- 使用inRange找到红色像素，使用bitwise_and用掩膜提取红色像素
- findContours找到的轮廓保存在vector类型的contours里面，绘制时要先创建一个与原图同大小的图像用于绘制
- 画bounding box：用for循环寻找

```CPP
Rect bounding_rect = boundingRect(contours[i]);
```

- 计算轮廓面积：由于轮廓太多，故太小的就不显示

```CPP
if (arcLength(contours[i], true) > 10 && area > 10)
```

## 提取高亮区域并处理
- 灰度化同前，二值化使用threshold
- 膨胀和腐蚀需要定义核
```CPP
Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
```
- 漫水使用floodfill函数，需要指定seedPoint（漫水起始点），和颜色

## 图像绘制
- 圆：给出圆心和半径
- 矩形：给出左上角和右下角
- 写文字：putText，给出起始位置，字体类型，字体大小，字体线宽
- 外轮廓和bounding box同前

## 图像处理
### 旋转
- 获取中心点
- 计算旋转矩阵：给出旋转角度和缩放因子
```CPP
Mat rotation_matrix = getRotationMatrix2D(rotated_center, angle, scale);
```
- 计算旋转后的图像大小并调整旋转矩阵以考虑平移
```CPP
Rect2f bbox = RotatedRect(rotated_center, src.size(), angle).boundingRect2f();
rotation_matrix.at<double>(0, 2) += bbox.width / 2.0 - center.x;
    rotation_matrix.at<double>(1, 2) += bbox.height / 2.0 - center.y;
```
- 最后用warpAffine旋转

### 裁减
- 给出区域roi，src(roi)即可

## 最终图片展示
![图像颜色空间转换](https://github.com/voldbaboon/voldbaboon-s-RMV-training/blob/main/OpenCV_Project/resources/%E5%9B%BE%E5%83%8F%E9%A2%9C%E8%89%B2%E7%A9%BA%E9%97%B4%E8%BD%AC%E6%8D%A2.png)
![应用各种滤波操作](https://github.com/voldbaboon/voldbaboon-s-RMV-training/blob/main/OpenCV_Project/resources/%E5%BA%94%E7%94%A8%E5%90%84%E7%A7%8D%E6%BB%A4%E6%B3%A2%E6%93%8D%E4%BD%9C.png)
![特征提取](https://github.com/voldbaboon/voldbaboon-s-RMV-training/blob/main/OpenCV_Project/resources/%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96.png)
![图像绘制](https://github.com/voldbaboon/voldbaboon-s-RMV-training/blob/main/OpenCV_Project/resources/%E5%9B%BE%E5%83%8F%E7%BB%98%E5%88%B6.png)
![图像处理](https://github.com/voldbaboon/voldbaboon-s-RMV-training/blob/main/OpenCV_Project/resources/%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86.png)
