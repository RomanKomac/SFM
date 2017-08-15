#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <vector>

typedef std::pair< int,int > intPair;
typedef std::pair< cv::Mat,cv::Mat > MatPair;
typedef std::pair< intPair,cv::Mat > intPMat;
typedef std::pair< intPair,MatPair > intPMatP;

typedef std::vector<cv::KeyPoint> Keypoints;
typedef std::vector<cv::Point2f>  Points2f;
typedef std::vector<cv::Point3f>  Points3f;
typedef std::vector<cv::Point2d>  Points2d;
typedef std::vector<cv::Point3d>  Points3d;

struct match2D3D_d {
    Points2d points2D;
    Points3d points3D;
};
struct match2D3D_f {
    Points2f points2D;
    Points3f points3D;
};

struct Features_d {
    Keypoints keyPoints;
    Points2d  points;
    cv::Mat   descriptors;
};
struct Features_f {
    Keypoints keyPoints;
    Points2f  points;
    cv::Mat   descriptors;
};

struct Mapped3DPoint_d {
    cv::Point3d p;
    std::map<int, int> views;
};
struct Mapped3DPoint_f {
    cv::Point3f p;
    std::map<int, int> views;
};

struct Mapped3DPointRGB_d {
    Mapped3DPoint_d p;
    cv::Scalar   rgb;
};

struct Mapped3DPointRGB_f {
    Mapped3DPoint_f p;
    cv::Scalar   rgb;
};

typedef std::vector<Mapped3DPoint_d>     PointCloud_d;
typedef std::vector<Mapped3DPoint_f>     PointCloud_f;
typedef std::vector<Mapped3DPointRGB_d>  RGBPointCloud_d;
typedef std::vector<Mapped3DPointRGB_f>  RGBPointCloud_f;