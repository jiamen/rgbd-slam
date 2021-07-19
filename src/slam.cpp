//
// Created by zlc on 2021/7/10.
//

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

#include "slamBase.h"

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>


// 把g2o的定义放到前面
typedef g2o::BlockSolver_6_3 SlamBlockSolver;
typedef g2o::LinearSolverEigen<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;
// 注意这里的求解方法是LinearSolverEigen，不是LinearSolverCSparse


// 给定index，读取一帧数据
FRAME readFrame(int index, ParameterReader& pd);
// 估计一个运动的大小
double normofTransform(cv::Mat rvec, cv::Mat tvec);

// 检测两个帧，结果定义
enum CHECK_RESULT {NOT_MATCHED=0, TOO_FAR_AWAY, TOO_CLOSE, KEYFRAME};
// 函数声明
CHECK_RESULT checkKeyframes(FRAME& f1, FRAME& f2, g2o::SparseOptimizer& opti, bool is_loops=false);

// 检测近距离的回环
void checkNearbyLoops(vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti);
// 随机检测回环
void checkRandomLoops(vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti);


int main(int argc, char* *argv)
{
    // 前面部分和vo是一样的
    ParameterReader pd;
    int startIndex = atoi( pd.getData("start_index").c_str() );
    int endIndex   = atoi( pd.getData("end_index"  ).c_str() );

    cout << "startIndex = " << startIndex << endl;
    cout << "endIndex = " << endIndex << endl;

    // 所有的关键帧都放在了这里
    vector< FRAME > keyframes;
    // initialize
    cout << "Initializing ..." << endl;

    int currIndex = startIndex;     // 当前索引为currIndex
    FRAME currFrame = readFrame(currIndex, pd);    // 上一帧数据

    string detector = pd.getData("detector");
    string descriptor = pd.getData("descriptor");
    CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();

    /*
     * 计算第一帧的关键点和描述子
     * */
    computeKeyPointsAndDesp(currFrame, detector, descriptor);

    PointCloud::Ptr cloud = image2PointCloud(currFrame.rgb, currFrame.depth, camera);

    /******************************
     * 新增：有关g2o的初始化
     */
    // 第1步：创建一个线性求解器LinearSolver
    SlamLinearSolver* linearSolver = new SlamLinearSolver();
    linearSolver->setBlockOrdering( false );
    // 第2步：创建BlockSolver，并用上面定义的线性求解器初始化
    SlamBlockSolver* blockSolver = new SlamBlockSolver( std::unique_ptr<SlamBlockSolver::LinearSolverType>(linearSolver) );
        // SlamBlockSolver* blockSolver = new SlamBlockSolver( linearSolver );
    // 第3步：创建总求解器solver，并从GN，LM，Dogleg中选一个，再用上述块求解器BlockSolver初始化
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<SlamBlockSolver>(blockSolver));
    // g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( blockSolver );

    // 第4步：创建稀疏优化器
    g2o::SparseOptimizer globalOptimizer;   // 最后用的就是这个东西
    globalOptimizer.setAlgorithm( solver );
    // 不要输出调试信息
    globalOptimizer.setVerbose(false);

    // 第5步：添加顶点和边
    // 向globalOptimizer增加第一个顶点
    g2o::VertexSE3* v = new g2o::VertexSE3();
    v->setId( currIndex );
    v->setEstimate(Eigen::Isometry3d::Identity());
    v->setFixed(true);      // 第一个顶点固定，不用优化
    globalOptimizer.addVertex( v );

    keyframes.push_back(currFrame);

    double keyframe_threshold = atof( pd.getData("keyframe_threshold").c_str() );   // 关键帧的阈值：0.1
    bool check_loop_closure = pd.getData("check_loop_closure")==string("yes");           // 检测回环

    for (currIndex=startIndex+1; currIndex<endIndex; currIndex ++)
    {
        cout << "Reading files " << currIndex << endl;
        FRAME currFrame = readFrame(currIndex, pd);     // 读取currFrame
        computeKeyPointsAndDesp(currFrame, detector, descriptor);   // 提取特征，计算出关键点和匹配描述子
        CHECK_RESULT result = checkKeyframes(keyframes.back(), currFrame, globalOptimizer); // 匹配该帧与keyframes里最后一帧
        switch (result)
        {
            case NOT_MATCHED:
                // 没匹配上，直接跳过
                cout << RED"Not enough inliers." << endl;
                break;
            case TOO_FAR_AWAY:
                // 太远了，可能出错了
                cout << RED"Too far away, may be an error." << endl;
                break;
            case TOO_CLOSE:
                // 太近了，直接跳过
                cout << RESET"Too close, not a keyframe." << endl;
                break;
            case KEYFRAME:
                // 不远不近，刚刚好
                cout << GREEN"This is a new keyframe." << endl;
                /*
                 * This is important!!!
                 * This is important!!!
                 * This is important!!!
                 * */
                // 检测回环
                if (check_loop_closure)
                {
                    checkNearbyLoops(keyframes, currFrame, globalOptimizer);
                    checkRandomLoops(keyframes, currFrame, globalOptimizer);
                }
                keyframes.push_back(currFrame);
                break;

            default:
                break;
        }
    }

    // 第6步：执行优化
    cout << RESET"optimizing pose graph, vertices: " << globalOptimizer.vertices().size() << endl;
    globalOptimizer.save("./result_before.g2o");
    globalOptimizer.initializeOptimization();
    globalOptimizer.optimize(100);          // 可以指定优化步数
    globalOptimizer.save("./result_after.g2o");
    cout << "Optimization done." << endl;


    /*
     * 拼接点云地图
     * */
    cout << "saving the point cloud map ..." << endl;
    PointCloud::Ptr output(new PointCloud());       // 全局地图
    PointCloud::Ptr tmp(new PointCloud());

    pcl::VoxelGrid<PointT> voxel;       // 网格滤波器，调整地图分辨率
    pcl::PassThrough<PointT> pass;      // z方向区间滤波器，由于rgbd相机的有效深度区间有限，把太远的去掉

    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.0, 4.0);     // 4m以上的就不要了

    double gridsize = atof(pd.getData("voxel_grid").c_str());   // 分辨图可以在parameters.txt里调
    voxel.setLeafSize(gridsize, gridsize, gridsize);

    for (size_t i=0; i<keyframes.size(); i ++)
    {
        // 从g2o里取出一帧
        g2o::VertexSE3* vertex = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex(keyframes[i].frameID));
        Eigen::Isometry3d pose = vertex->estimate();        // 该帧优化后的位姿
        PointCloud::Ptr newCloud = image2PointCloud(keyframes[i].rgb, keyframes[i].depth, camera);  // 转成点云

        // 以下是两种滤波操作，分别是网格滤波器和区间滤波
        voxel.setInputCloud(newCloud);
        voxel.filter(*tmp);
        pass.setInputCloud(tmp);
        pass.filter(*newCloud);

        // 把点云变换后加入到全局地图中
        pcl::transformPointCloud(*newCloud, *tmp, pose.matrix());
        *output += *tmp;

        tmp->clear();
        newCloud->clear();
    }

    voxel.setInputCloud(output);
    voxel.filter(*tmp);


    // 存储
    pcl::io::savePCDFile("./result.pcd", *tmp);

    cout << "Final map is saved." << endl;

    return 0;
}


// 读取一帧rgb图像和对应的深度depth图
FRAME readFrame(int index, ParameterReader& pd)
{
    FRAME f;
    string rgbDir   = pd.getData("rgb_dir");
    string depthDir = pd.getData("depth_dir");

    string rgbExt   = pd.getData("rgb_extension");      // .png
    string depthExt = pd.getData("depth_extension");    // .png

    stringstream ss;
    ss << rgbDir << index << rgbExt;
    string filename;
    ss >> filename;
    f.rgb = cv::imread(filename);

    ss.clear();
    filename.clear();
    ss << depthDir << index << depthExt;
    ss >> filename;

    f.depth = cv::imread(filename, -1);
    f.frameID = index;

    return f;
}



double normofTransform(cv::Mat rvec, cv::Mat tvec)
{
    return fabs( min(cv::norm(rvec), 2*M_PI - cv::norm(rvec)) ) + fabs(cv::norm(tvec));
}


// 检测是否为关键帧，
CHECK_RESULT checkKeyframes(FRAME& f1, FRAME& f2, g2o::SparseOptimizer& opti, bool is_loops)
{
    static ParameterReader pd;
    static int min_inliers = atoi( pd.getData("min_inliers").c_str() );
    static double max_norm = atof( pd.getData("max_norm").c_str() );
    static double keyframe_threshold = atof( pd.getData("keyframe_threshold").c_str() );
    static double max_norm_lp = atof( pd.getData("max_norm_lp").c_str() );
    static CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();

    // 比较f1和f2，对两帧进行暴力匹配，挑选出好的匹配点，对第一个匹配帧关键点进行2D=》3D转换，对第二个匹配帧关键点直接存储，构建PnP问题并求解，得到运动估计
    RESULT_OF_PNP result = estimateMotion(f1, f2, camera);
    if ( result.inliers < min_inliers )     // inliers不够，放弃该帧
        return NOT_MATCHED;

    // 计算运动范围是否太大
    double norm = normofTransform( result.rvec, result.tvec );
    if (is_loops == false)
    {
        if (norm >= max_norm)
            return TOO_FAR_AWAY;            // too far away, may be error
    }
    else
    {
        if (norm >= max_norm_lp)
            return TOO_FAR_AWAY;
    }
    if (norm <= keyframe_threshold)
        return TOO_CLOSE;


    // 向g2o中增加这个顶点与上一帧联系的边
    // ① 顶点部分：顶点只需设定id即可
    if (is_loops == false)
    {
        g2o::VertexSE3* v = new g2o::VertexSE3();
        v->setId(f2.frameID);
        v->setEstimate(Eigen::Isometry3d::Identity());
        opti.addVertex(v);
    }

    // ② 边部分
    g2o::EdgeSE3* edge = new g2o::EdgeSE3();
    edge->setVertex(0, opti.vertex(f1.frameID));    // 连接此边的两个顶点id
    edge->setVertex(1, opti.vertex(f2.frameID));
    edge->setRobustKernel( new g2o::RobustKernelHuber() );

    // 信息矩阵
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();

    // 信息矩阵是协方差矩阵的逆，表示我们对边的精度的预先估计
    // 因此pose为6D的，信息矩阵是6*6的矩阵，假设位置和角度的估计精度均为0.1且互相独立
    // 那么协方差则为对角为0.01的矩阵，信息阵则为100的矩阵
    information(0, 0) = information(1, 1) = information(2, 2) = 100;
    information(3, 3) = information(4, 4) = information(5, 5) = 100;

    // 也可以将角度设大一些，表示对角度的估计更加准确
    edge->setInformation(information);

    // 边的估计即是 pnp求解的结果
    Eigen::Isometry3d T = cvMat2Eigen(result.rvec, result.tvec);
    // edge->setMeasurement( T );
    edge->setMeasurement(T.inverse());

    // 将此边加入图中
    opti.addEdge(edge);
    return KEYFRAME;
}


// 检测近距离的回环
void checkNearbyLoops(vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti)
{
    static ParameterReader pd;
    static int nearby_loops = atoi(pd.getData("nearby_loops").c_str()); // 5

    // 就是把currFrame 和 frames里末尾几个测一遍
    if ( frames.size() <= nearby_loops )
    {
        // no enough keyframes, check everyone
        for (size_t i=0; i<frames.size(); i ++)
        {
            checkKeyframes(frames[i], currFrame, opti, true);
        }
    }
    else
    {
        // check the nearest ones
        for (size_t i=frames.size()-nearby_loops; i<frames.size(); i ++)
        {
            checkKeyframes(frames[i], currFrame, opti, true);
        }
    }
}

// 随机检测回环
void checkRandomLoops(vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti)
{
    static ParameterReader pd;
    static int random_loops = atoi(pd.getData("random_loops").c_str());
    srand((unsigned int)time(NULL));

    // 随机取一些帧进行检测
    if (frames.size() <= random_loops)
    {
        // no enough keyframes, check everyone
        for (size_t i=0; i<frames.size(); i ++)
        {
            checkKeyframes(frames[i], currFrame, opti, true);
        }
    }
    else
    {
        // randomly check loops
        for (int i=0; i<random_loops; i ++)
        {
            int index = rand() % frames.size();
            checkKeyframes(frames[index], currFrame, opti, true);
        }
    }
}

