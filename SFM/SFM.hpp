#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string>
#include <vector>
#include "typedefs.hpp"
#include "bundle_adjustment/ba.hpp"

#pragma once

class stereoMultihold
{
	public:
		intPair img_index;
		cv::Mat F;
		cv::Mat E;
		cv::Mat R, t;
		cv::Mat mask;
		cv::DMatch matches;
};

class SFM
{
	public:
		SFM(std::vector<cv::Mat> images, cv::Mat K1, cv::Mat K2, std::vector<double> distort1, std::vector<double> distort2);
		SFM(std::vector<cv::Mat> images, cv::Mat K1, cv::Mat K2, std::vector<double> distort);
		SFM(std::vector<cv::Mat> images, cv::Mat K1, std::vector<double> distort);
		SFM(std::vector<cv::Mat> images, cv::Mat K1, cv::Mat K2);
		SFM(std::vector<cv::Mat> images, cv::Mat K1);
		SFM(std::vector<cv::Mat> images);
		void detect(cv::Ptr<cv::Feature2D> detector);
		void extract(cv::Ptr<cv::Feature2D> extractor);
		void match(cv::DescriptorMatcher &matcher, int match_method, int nmatches);
		void RANSACfundamental(double reprError, double confidence, int method, int min_inliers);
		void motionFromFundamental();
		void initialBundleAdjustment();
		void additionalBundleAdjustment();
		void mergePointCloud(const PointCloud_d& cloud);
		void visualizeCameraMotion();
		void denseReconstruction(int method);
		void showEpipolarLines();
		void showCorrespondences();
		void showTriangulation(int i, int j, float max_dist);
		int avg_num_iters;
		float avg_runtime;
	private:
		std::map<int,int> dependencyNet();
		void incrementalBundleAdjustment();
		static bool sortMatches(cv::DMatch m1, cv::DMatch m2){
			return m1.distance < m2.distance;
		}
		static bool sortMasks(std::pair< intPair, cv::Mat > m1, std::pair< intPair, cv::Mat > m2){
			return ((float)countNonZero(m1.second))/((float)m1.second.total()) > ((float)countNonZero(m2.second))/((float)m2.second.total());
		}
		std::vector<cv::Mat> GRAY_imgs;
		std::vector<cv::Mat> BGR_imgs;
		std::vector< std::vector< cv::KeyPoint > > keypoints;
		std::vector< cv::Mat > descriptors;
		std::vector< cv::Mat > masks;
		std::vector< std::vector< cv::DMatch > > matches;

		std::vector< std::pair< intPair, cv::Mat > > sortedMasks;
		std::map< intPair,MatPair > RtMap;
		std::map< intPair,cv::Mat > essenMap;
		std::map< intPair,cv::Mat > fundMap;
		std::map< intPair,cv::Mat > maskMap;
		std::map< intPair,std::vector< cv::DMatch > > matchMap;
		cv::Mat K1;
		cv::Mat K2;
		bool calibrated;
		cv::Size defImageSize;
		std::vector<double> distort1;
		std::vector<double> distort2;
		std::vector< Features_d > mImageFeatures;
		PointCloud_d             mReconstructionCloud;
		std::vector<cv::Matx34d> mCameraPoses;
		std::set<int>            mDoneViews;
    	std::set<int>            mGoodViews;
    	BundleAdjustment ba;
};