#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <string>
#include <vector>

// definitions for RANSAC variations
#define SFM_RANSAC 1
#define SFM_PE_RANSAC 2
#define SFM_PROSAC 3
#define SFM_MLESAC 4
#define SFM_ARRSAC 5

class FundMatEstimator
{
	protected:
		double reprojectError;
		double confidence;
		double inlier_ratio;
		std::vector<int> selection;
		virtual void subselect(cv::_InputArray _points1, cv::_InputArray _points2, cv::_OutputArray _output1, cv::_OutputArray _output2, std::vector<float> similarities) = 0;
	public:
		virtual int run(cv::_InputArray _points1, cv::_InputArray _points2, cv::_OutputArray F, cv::_OutputArray _mask, std::vector<float> similarities) = 0;
};

class Estimator
{
	public:
		static cv::Mat estFundamentalMat( cv::_InputArray _points1, cv::_InputArray _points2,
                                int method, double param1, double param2, 
                                cv::_OutputArray _mask, std::vector<float> similarities);
		static FundMatEstimator* createFundMatEstimator(int method, double param1, double param2);
		static int getInliers(cv::_InputArray _points1, cv::_InputArray _points2, cv::Mat _fundamentalMat, double err, cv::_OutputArray _mask);
		static int fundMat(cv::_InputArray _points1, cv::_InputArray _points2, cv::_OutputArray _F);
		static void debug(cv::_InputArray _points1, cv::_InputArray _points2, cv::_OutputArray _F);
		static int updateNumIters(double p, double ep, int modelPoints, int maxIters);
};