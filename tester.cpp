#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "SFM/constants.hpp"
#include "SFM/SFM.hpp"
#include "SFM/Image.hpp"

using namespace cv;
using namespace std;
using namespace xfeatures2d;

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

	SFM pipeline(Image::loadFromFolder(path, pattern), Mat(Matx33d(2892.3,0,823.21,0,2883.2,619.07,0,0,1)));

	// Finds and extracts keypoints
	#if defined VERBOSE
	cout << "Detecting and extracting keypoints" << endl;
	#endif	

	cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();

	pipeline.detect(f2d);
	pipeline.extract(f2d);

	// Matches and filters descriptors
	#if defined VERBOSE
	cout << "Matching and filtering descriptors" << endl;
	#endif	
	
	BFMatcher matcher;
	pipeline.match(matcher, MATCH_CONSECUTIVE, 100);


	// Estimates fundamental matrices using RANSAC
	#if defined VERBOSE
	cout << "Estimating fundamental matrices" << endl;
	#endif	

	// Reprojection error, the tolerance of the estimated inlier model
	double reprError = 1;
	// COnfidence of the model. 0.99 means 99% probability of it being correct.
	double confidence = 0.99;

	pipeline.RANSACfundamental(reprError, confidence, SFM_LO_RANSAC, 50);

	cout << "Average number of iterations: " << pipeline.avg_num_iters << endl;
	cout << "Average runtime: " << pipeline.avg_runtime << endl;

	// Estimates essential matrices from fundamental using intrinsic camera parameters
	// Estimated motion is used later in refining camera position and orientation
	#if defined VERBOSE
	cout << "Estimating essential matrices" << endl;
	#endif	

	pipeline.motionFromFundamental();


	//For every three views get points
	#if defined VERBOSE
	cout << "Calculating consistencies" << endl;
	#endif	

	//pipeline.buildNet();
	pipeline.minThreeViewsConsistency();
	pipeline.triangulationMinThreeViews();

	//pipeline.bundleAdjustment();
	pipeline.sparseReconstruction();
	
	// Adjusts bundle using Ceres solver
	#if defined VERBOSE
	cout << "Performing initial bundle adjustment" << endl;
	#endif	

	//pipeline.buildNet();
	cout << "after buildnet" << endl;
	//pipeline.initialBundleAdjustment();
	//pipeline.additionalBundleAdjustment();

	// Incremental bundle adjustment. Rotation and translation are pre-adjusted using PnP solver
	#if defined VERBOSE
	cout << "Performing additional bundle adjustment" << endl;
	#endif	

	//pipeline.incrementalBundleAdjustment();
	
	
	// Shows correspondences
	#if defined VERBOSE
	cout << "Camera motion visualization" << endl;
	#endif	

	//pipeline.showCorrespondences();
	pipeline.visualizeCameraMotion();
	//pipeline.showTriangulation(0,1);
  	
}