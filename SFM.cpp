#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "SFM/Image.hpp"
#include "SFM/RANSAC/ransac.hpp"

using namespace cv;
using namespace std;

void SFMPipeline(string path, string pattern);
bool sortMatches(DMatch m1, DMatch m2);

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

bool sortMatches(DMatch m1, DMatch m2){
	return m1.distance < m2.distance;
}

void SFMPipeline(string path, string pattern){
	// Loads images into an array
	vector<Mat> imgs;
	imgs = Image::loadFromFolder(path, pattern);


	#if defined VERBOSE
	cout << "Converting images to grayscale" << endl;
	#endif	
	
	// Convert images to grayscale
	for(int i = 0; i < imgs.size(); i++){
		Mat greyMat;
		cvtColor(imgs[i], greyMat, COLOR_BGR2GRAY);
		imgs[i] = greyMat;

	}

	#if defined VERBOSE
	cout << "Extracting keypoints" << endl;
	#endif	
	// Finds and extracts keypoints
	vector< vector< KeyPoint > > keypoints;
	vector< Mat > descriptors;
	
	SiftFeatureDetector detector;
	SurfDescriptorExtractor extractor;

	for(int i = 0; i < imgs.size(); i++){
		vector< KeyPoint > kpts;
		Mat desc;
		detector.detect(imgs[i], kpts);
		extractor.compute(imgs[i], kpts, desc);
		keypoints.push_back(kpts);
		descriptors.push_back(desc);
	}


	#if defined VERBOSE
	cout << "Matching and filtering descriptors" << endl;
	#endif	
	// Matches descriptors of image pairs
	// In this step we assume consecutiveness of pairs
	BFMatcher matcher;
	vector< vector< DMatch > > matches;

	// Rule of thumb proposed in Lowe's paper
	// If the (strength_first_match < ratio*strength_second_match) then keep
	const float ratio = 0.8;

	for(int i = 1; i < imgs.size(); i++){
		vector< vector< DMatch > > localMatches;
		vector< DMatch > goodMatches;
		matcher.knnMatch(descriptors[i-1], descriptors[i], localMatches, 2);
		for (int j = 0; j < localMatches.size(); j++)
		{
		    if (localMatches[j][0].distance < ratio * localMatches[j][1].distance)
		    {
		        goodMatches.push_back(localMatches[j][0]);
		    }
		}
		sort(goodMatches.begin(), goodMatches.end(), sortMatches); 
		matches.push_back(goodMatches);
	}

	// Reprojection error, the tolerance of the estimated inlier model
	double reprError = 2;
	// COnfidence of the model. 0.99 means 99% probability of it being correct.
	double confidence = 0.99;

	for(int i = 0; i < matches.size(); i++){
		vector<Point2f> points1;
		vector<Point2f> points2;
		vector<float> similarities;
		int m = (matches[i].size());
		for(int k = 0; k < m; k++){
			points1.push_back(keypoints[0][matches[i][k].queryIdx].pt);
			points2.push_back(keypoints[1][matches[i][k].trainIdx].pt);
			similarities.push_back(matches[i][k].distance);
		}
		
		Mat mask1;
		Mat F;

		//Default implemented RANSAC fundamental estimator
		//F = findFundamentalMat(points1, points2, FM_RANSAC, 2, 0.99, mask1 );

		//RANSAC
		F = Estimator::estFundamentalMat(points1, points2, SFM_RANSAC, reprError, confidence, mask1);
		
		//RANSAC with Td,d test
		//F = Estimator::estFundamentalMat(points1, points2, SFM_RANSAC_TddTest, reprError, confidence, mask1);

		//RANSAC with Td,d test using similarity score
		//F = Estimator::estFundamentalMat(points1, points2, SFM_RANSAC_TddTest_SScore, reprError, confidence, mask1, similarities);

		//Preemptive RANSAC
		//F = Estimator::estFundamentalMat(points1, points2, SFM_PE_RANSAC, reprError, confidence, mask1);

		//PROSAC
		//F = Estimator::estFundamentalMat(points1, points2, SFM_PROSAC, reprError, confidence, mask1, similarities);

		//MLESAC
		//F = Estimator::estFundamentalMat(points1, points2, SFM_MLESAC, reprError, confidence, mask1, similarities);

		//ARRSAC
		//F = Estimator::estFundamentalMat(points1, points2, SFM_ARRSAC, reprError, confidence, mask1, similarities);

		cout << F << endl;
		//int numm = countNonZero(mask2);
		//vector<Point2f> pointsbet1;
		//vector<Point2f> pointsbet2;

		vector<DMatch> good_matches;
		for(int l = 0; l < m; l++){
			if(mask1.at<unsigned char>(l)){
				good_matches.push_back(matches[i][l]);
				//pointsbet1.push_back(points1[l]);
				//pointsbet2.push_back(points2[l]);
			}
		}



		Mat img_matches;
		drawMatches( imgs[i], keypoints[i], imgs[i+1], keypoints[i+1],
	               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
	               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


		Size size(img_matches.cols/2, img_matches.rows/2);
		Mat dst;//dst image
		resize(img_matches,dst,size);
	  	//-- Show detected matches
	  	imshow( "Good Matches", dst );
	  	waitKey(-1);
	}
  
}