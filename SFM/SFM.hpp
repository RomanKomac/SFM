#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <string>
#include <vector>

class SFM
{
	public:
		SFM(std::vector<cv::Mat> images);
		void detect(cv::FeatureDetector &detector);
		void extract(cv::DescriptorExtractor &extractor);
		void extract(cv::FREAK &extractor);
		void match(cv::DescriptorMatcher &matcher);
		void RANSACfundamental(double reprError, double confidence, int method);
		void showCorrespondences();
		int avg_num_iters;
		float avg_runtime;
	private:
		static bool sortMatches(cv::DMatch m1, cv::DMatch m2){
			return m1.distance < m2.distance;
		}
		std::vector<cv::Mat> GRAY_imgs;
		std::vector<cv::Mat> BGR_imgs;
		std::vector< std::vector< cv::KeyPoint > > keypoints;
		std::vector< cv::Mat > descriptors;
		std::vector< cv::Mat > masks;
		std::vector< cv::Mat > fundMatrices;
		std::vector< std::vector< cv::DMatch > > matches;

};