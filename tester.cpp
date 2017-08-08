#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "SFM/constants.hpp"
#include "SFM/SFM.hpp"
#include "SFM/Image.hpp"

using namespace cv;
using namespace std;

void SFMPipeline(string path, string pattern);

int main(int argc, char** argv )
{
	if(argc >= 3){
		SFMPipeline(argv[1], argv[2]);
	}
	else if (argc >= 2){
    	SFMPipeline(argv[1], "");

	} else {
		cout << "USAGE" << endl;
		cout << "./StructureFromMotion arg1 arg2 ..." << endl;
		cout << "arg1 must be a directory that contains ImageList.txt" << endl;
		cout << "arg2 is optional, if included should be general image name regex(\%02d.jpg)";
		cout << "ImageList.txt should contain lines of relative image paths." << endl;
		cout << "Other arguments are currently optional" << endl;
	}
	

    return 0;
}

void SFMPipeline(string path, string pattern){
	
	// Loads images
	#if defined VERBOSE
	cout << "Loading images" << endl;
	#endif

	SFM pipeline(Image::loadFromFolder(path, pattern));

	// Finds and extracts keypoints
	#if defined VERBOSE
	cout << "Detecting and extracting keypoints" << endl;
	#endif	
	
	SiftFeatureDetector detector;
	SiftDescriptorExtractor extractor;
	//DescriptorExtractor extractor = new ;

	pipeline.detect(detector);
	pipeline.extract(extractor);

	// Matches and filters descriptors
	#if defined VERBOSE
	cout << "Matching and filtering descriptors" << endl;
	#endif	
	
	BFMatcher matcher;
	pipeline.match(matcher);


	// Estimates fundamental matrices using RANSAC
	#if defined VERBOSE
	cout << "Estimating fundamental matrices" << endl;
	#endif	

	// Reprojection error, the tolerance of the estimated inlier model
	double reprError = 1;
	// COnfidence of the model. 0.99 means 99% probability of it being correct.
	double confidence = 0.999;

	pipeline.RANSACfundamental(reprError, confidence, SFM_RANSAC);

	// Shows correspondences
	#if defined VERBOSE
	cout << "Showing correspondences" << endl;
	#endif	

	pipeline.showCorrespondences();
  
}