#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <string>
#include <vector>
#include "../constants.hpp"
#include "../typedefs.hpp"
//Benchmarking purposes
#include "time.h"

//Include only once, to avoid redefinitions
#pragma once

class FundMatEstimator
{
	protected:
		double reprojectError;
		double confidence;
		double inlier_ratio;
		int num_iters;
		float loop_runtime;
		std::vector<int> selection;
	public:
		virtual int run(cv::_InputArray _points1, cv::_InputArray _points2, cv::_OutputArray F, cv::_OutputArray _mask, std::vector<float> similarities) = 0;
		virtual int lastNumOfIterations(){return num_iters;};
		virtual float lastRuntime(){return loop_runtime;};
		int M, B, D, INNER_LOOP;
};

class Estimator
{
	public:
		static int num_iters;
		static float runtime;
		static std::vector<int> pool;
		static cv::Mat estFundamentalMat( cv::_InputArray _points1, cv::_InputArray _points2,
                                int method, double param1, double param2, double param3,
                                cv::_OutputArray _mask, std::vector<float> similarities);
		static cv::Mat estFundamentalMat( cv::_InputArray _points1, cv::_InputArray _points2,
                                int method, double param1, double param2,
                                cv::_OutputArray _mask, std::vector<float> similarities);
		static cv::Mat estFundamentalMat( cv::_InputArray _points1, cv::_InputArray _points2,
                                int method, double param1, double param2, double param3,
                                cv::_OutputArray _mask);
		static cv::Mat estFundamentalMat( cv::_InputArray _points1, cv::_InputArray _points2,
                                int method, double param1, double param2,
                                cv::_OutputArray _mask);
		static cv::Mat triangulateDLT(cv::_InputArray _P1, cv::_InputArray _P2, cv::_InputArray _p1, cv::_InputArray _p2);
		static bool triangulateViews(std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2, 
                                 cv::Matx34f Pleft, cv::Matx34f Pright, std::vector<intPair> indxs, intPair pr, cv::Mat K, PointCloud_f& pointCloud);
		static cv::Mat essentialFromFundamental(cv::_InputArray _F, cv::_InputArray _K1, cv::_InputArray _K2);
		static bool motionFromEssential(cv::_InputArray _E, cv::_InputArray _K1, cv::_InputArray _K2, cv::_InputArray _p1, cv::_InputArray _p2, cv::_OutputArray _R, cv::_OutputArray _T);
		static FundMatEstimator* createFundMatEstimator(int method, double param1, double param2, double param3);
		static int getInliers(cv::_InputArray _points1, cv::_InputArray _points2, cv::Mat _fundamentalMat, double err, cv::_OutputArray _mask = cv::noArray());
		static int fundMat(cv::_InputArray _points1, cv::_InputArray _points2, cv::_OutputArray _F, bool useAll);
		static void debug(cv::_InputArray _points1, cv::_InputArray _points2, cv::_OutputArray _F);
		static int updateNumIters(double p, double ep, int modelPoints, int maxIters);
		static void subselect(cv::_InputArray _points1, cv::_InputArray _points2, cv::_OutputArray _output1, cv::_OutputArray _output2, int len, int limit = 0);
};