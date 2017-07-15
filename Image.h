#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class Image
{
	public:
		static std::vector<cv::Mat> loadFromFolder(std::string path, std::string reg_name);

};