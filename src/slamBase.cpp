//
// Created by zlc on 2021/7/10.
//
#include "slamBase.h"

using namespace cv;

// 将RGB图像像素转换为点云
PointCloud::Ptr image2PointCloud(cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera)
{
    PointCloud::Ptr cloud(new PointCloud);

    // 注意：Mat访问都是行优先
    for (int m=0; m<depth.rows; m += 2)
        for (int n=0; n<depth.cols; n += 2)
        {
            // 获取深度图中(m, n)处的值
            ushort d = depth.ptr<ushort>(m)[n];
            // d可能没有值，若如此，跳过此点
            if (d == 0)
                continue;
            // d存在值，则向点云中增加一个点
            PointT p;

            // 计算这个点的空间坐标
            p.z = double(d) / camera.scale;
            p.x = (n - camera.cx) * p.z / camera.fx;
            p.y = (m - camera.cy) * p.z / camera.fy;

            // 从rgb图像中获取它的颜色
            // rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
            p.b = rgb.ptr<uchar>(m)[n*3];
            p.g = rgb.ptr<uchar>(m)[n*3+1];
            p.r = rgb.ptr<uchar>(m)[n*3+2];

            // 把p加入到点云中
            cloud->points.push_back( p );
        }
    // 设置并保存点云
    cloud->height = 1;
    cloud->width  = cloud->points.size();
    cloud->is_dense = false;

    return cloud;
}

cv::Point3f point2dTo3d(cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera)
{
    cv::Point3f p;  // 3D点

    p.z = double(point.z) / camera.scale;
    p.x = (point.x - camera.cx) * p.z / camera.fx;
    p.y = (point.y - camera.cy) * p.z / camera.fy;

    return p;
}


// computeKeyPointsAndDesp  同时提取关键点与特征点描述子
void computeKeyPointsAndDesp(FRAME& frame, string detector, string descriptor)
{
    cv::Ptr<cv::FeatureDetector> _detector;
    cv::Ptr<cv::DescriptorExtractor> _descriptor;

    if (detector == "ORB")
    {
        _detector = cv::ORB::create();
    }
    if (descriptor == "ORB")
    {
        _descriptor = cv::ORB::create();
    }
    if (!_detector || !_descriptor)
    {
        cerr << "Unknown detector or descriptor type !" << detector << ", " <<descriptor << endl;
        return ;
    }

    // 第一步：检测Oriented FAST角点位置
    _detector->detect(frame.rgb, frame.kp);
    // 第二步：根据角点位置计算BRIEF描述子
    _descriptor->compute(frame.rgb, frame.kp, frame.desp);

    return ;
}


// estimateMotion 使用3d-2dPnP来，计算两个帧之间的运动
// 输入：帧1和帧2
// 输出：rvec 和 tvec
RESULT_OF_PNP estimateMotion(FRAME& frame1, FRAME& frame2, CAMERA_INTRINSIC_PARAMETERS& camera)
{
    static ParameterReader pd;
    vector<cv::DMatch> matches;     // 存储匹配点

    // 第1步：对两帧的特征点使用Hamming距离进行暴力匹配
    cv::BFMatcher matcher;
    matcher.match(frame1.desp, frame2.desp, matches);

    // 第2步：匹配点对筛选：找出最小匹配距离，并进行数量检查
    RESULT_OF_PNP result;           // 待返回值
    vector<cv::DMatch> goodMatches; // 存储好的匹配点
    double minDis = 9999;
    double good_match_threshold = atof(pd.getData("good_match_threshold").c_str());
    for (size_t i=0; i<matches.size(); i ++)        // 找出所有匹配点对中的最小距离，即醉相思的两组点之间的距离，作为后续筛选的阈值
    {
        if (matches[i].distance < minDis)
            minDis = matches[i].distance;
    }
    if (minDis < 10)        // 描述子匹配最小距离，即最相似的点
        minDis = 10;

    for (size_t i=0; i<matches.size(); i ++)
    {
        // if (matches[i].distance < max(30.0, good_match_threshold*minDis))      // 工程经验
        if (matches[i].distance < good_match_threshold*minDis)      // 工程经验
            goodMatches.push_back(matches[i]);
    }
    if (goodMatches.size() <= 5)
    {
        result.inliers = -1;
        return result;
    }


    // 第3步：对筛选出的特征点进行PnP求解问题构造：  （PnP是 求解3D点到2D点对运动 的方法）
    vector<cv::Point3f> pts_obj;    // 第一帧的三维点
    vector<cv::Point2f> pts_img;    // 第二帧的图像点

    for (size_t i=0; i<goodMatches.size(); i ++)        // 需要相机内参
    {
        // ①第1个匹配点转换为相机坐标系下的3维空间点    query是第一个，train是第二个
        cv::Point2f p = frame1.kp[goodMatches[i].queryIdx].pt;
        // 获取 深度d 时要小心！！！ x是向右的列，y是向下的行
        ushort d = frame1.depth.ptr<ushort>( int(p.y) )[int(p.x)];
        if (d == 0)
            continue;
        // 将(u,v,d)转成(x,y,z)
        cv::Point3f pt(p.x, p.y, d);
        cv::Point3f pd = point2dTo3d(pt, camera);
        pts_obj.push_back(pd);

        // ②第2个匹配点直接存
        pts_img.push_back( cv::Point2f(frame2.kp[goodMatches[i].trainIdx].pt) );
    }
    if (pts_obj.size() <= 5 || pts_img.size() <= 5)
    {
        result.inliers = -1;
        return result;
    }

    // 相机内参矩阵
    double camera_matrix_data[3][3] = {
            {camera.fx, 0, camera.cx},
            {0, camera.fy, camera.cy},
            {0, 0, 1}
    };
    cv::Mat cameraMatrix(3, 3, CV_64F, camera_matrix_data);     // 构建相机矩阵，cv::Mat类型的矩阵
    cv::Mat rvec, tvec, inliers;

    // 第4步：求解PnP    上一帧下的三维点  本帧下的二维点                        无初始估计值，这里是两帧之间的变换
    cv::solvePnPRansac(pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, 0.99, inliers);

    result.rvec = rvec;     // 求得的是R_21
    result.tvec = tvec;
    result.inliers = inliers.rows;      // 记录内点数量

    return result;
}


// cvMat 2 Eigen  注意：这里解析出的变换矩阵是 T_21， 即第一帧转换到第二帧
Eigen::Isometry3d cvMat2Eigen(cv::Mat& rvec, cv::Mat& tvec)
{
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    Eigen::Matrix3d r;
    for (int i=0; i<3; i ++)
    {
        for (int j=0; j<3; j ++)
            r(i, j) = R.at<double>(i, j);
    }

    // 将平移向量 和 旋转矩阵 转换成 变换矩阵
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

    Eigen::AngleAxisd angle(r);     // 直接使用旋转矩阵来对旋转向量赋值
    T = angle;                      // “=”重载，使用旋转向量初始化变换矩阵旋转矩阵部分
    T(0, 3) = tvec.at<double>(0, 0);        // 矩阵右上角的元素，这里开始平移部分
    T(1, 3) = tvec.at<double>(1, 0);
    T(2, 3) = tvec.at<double>(2, 0);

    return T;
}

// jointPointCloud
// 输入：原始点云，新来的帧以及它的位姿
// 输出：将新来帧加到原始帧后的图像
PointCloud::Ptr jointPointCloud(PointCloud::Ptr original, FRAME& newFrame, Eigen::Isometry3d T, CAMERA_INTRINSIC_PARAMETERS& camera)
{
    PointCloud::Ptr newCloud = image2PointCloud(newFrame.rgb, newFrame.depth, camera);

    // 合并点云
    PointCloud::Ptr output(new PointCloud());
    pcl::transformPointCloud(*original, *output, T.matrix());       // 将原始点云变换到当前帧坐标系下
    *newCloud += *output;                   // 拼接融合


    // Voxel grid 滤波降采样
    static pcl::VoxelGrid<PointT> voxel;
    static ParameterReader pd;

    double gridsize = atof(pd.getData("voxel_grid").c_str());
    voxel.setLeafSize(gridsize, gridsize, gridsize);
    voxel.setInputCloud( newCloud );
    PointCloud::Ptr tmp( new PointCloud() );        // 滤波后的点云
    voxel.filter(*tmp);

    return tmp;
}



