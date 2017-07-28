#include <opencv2/opencv.hpp>
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
	private:
		double reprojectError;
		double confidence;
	public:
		virtual int run(cv::Mat m1, cv::Mat m2, cv::Mat F) = 0;
};

class Estimator
{
	public:
		static cv::Mat estFundamentalMat( cv::_InputArray _points1, cv::_InputArray _points2,
                                int method, double param1, double param2, 
                                cv::_OutputArray _mask, std::vector<float> similarities);
		static FundMatEstimator* createFundMatEstimator(int method, double param1, double param2, cv::_OutputArray _mask, std::vector<float> similarities);
};