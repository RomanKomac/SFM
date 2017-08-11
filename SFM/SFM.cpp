#include "SFM.hpp"
#include "estimator/estimator.hpp"

using namespace std;
using namespace cv;

SFM::SFM(vector<Mat> images, Mat _K1, Mat _K2){
	BGR_imgs = images;
	GRAY_imgs = vector<Mat>();

	K1 = _K1;
	K2 = _K2;

	for(int i = 0; i < images.size(); i++){
		Mat greyMat;
		cvtColor(images[i], greyMat, COLOR_BGR2GRAY);
		GRAY_imgs.push_back(greyMat);
	}
}

void SFM::detect(FeatureDetector &detector){
	for(int i = 0; i < GRAY_imgs.size(); i++){
		vector< KeyPoint > kpts;
		detector.detect(GRAY_imgs[i], kpts);
		keypoints.push_back(kpts);
	}
}

void SFM::extract(DescriptorExtractor &extractor){
	for(int i = 0; i < GRAY_imgs.size(); i++){
		Mat desc;
		extractor.compute(GRAY_imgs[i], keypoints[i], desc);
		descriptors.push_back(desc);
	}
}

void SFM::extract(FREAK &extractor){
	for(int i = 0; i < GRAY_imgs.size(); i++){
		Mat desc;
		extractor.compute(GRAY_imgs[i], keypoints[i], desc);
		descriptors.push_back(desc);
	}
}

void SFM::extract(SIFT &extractor){
	for(int i = 0; i < GRAY_imgs.size(); i++){
		Mat desc;
		extractor.compute(GRAY_imgs[i], keypoints[i], desc);
		descriptors.push_back(desc);
	}
}

void SFM::extract(SURF &extractor){
	for(int i = 0; i < GRAY_imgs.size(); i++){
		Mat desc;
		extractor.compute(GRAY_imgs[i], keypoints[i], desc);
		descriptors.push_back(desc);
	}
}

void SFM::match(DescriptorMatcher &matcher){
	// Rule of thumb proposed in Lowe's paper
	// If the (strength_first_match < ratio*strength_second_match) then keep
	const float ratio = 0.8;

	for(int i = 1; i < GRAY_imgs.size(); i++){
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
}

void SFM::RANSACfundamental(double reprError, double confidence, int method){
	fundMatrices = vector<Mat>(matches.size());
	masks = vector<Mat>(matches.size());

	avg_num_iters = 0;
	avg_runtime = 0;

	for(int i = 0; i < matches.size(); i++){
		vector<Point2f> points1;
		vector<Point2f> points2;
		vector<float> similarities;
		for(int k = 0; k < matches[i].size(); k++){
			points1.push_back(keypoints[i][matches[i][k].queryIdx].pt);
			points2.push_back(keypoints[i+1][matches[i][k].trainIdx].pt);
			similarities.push_back(matches[i][k].distance);
		}
		
		Mat mask;
		Mat F;

		F = Estimator::estFundamentalMat(points1, points2, method, reprError, confidence, mask, similarities);
		fundMatrices[i] = F;
		masks[i] = mask;
		
		avg_num_iters += Estimator::num_iters;
		avg_runtime += Estimator::runtime;
	}
	avg_num_iters /= matches.size();
	avg_runtime /= matches.size();
}

void SFM::motionFromFundamental(){
	essenMatrices = vector<Mat>(matches.size());
	RtMatrices = vector< pair<Mat,Mat> >(matches.size());

	for(int i = 0; i < matches.size(); i++){
		vector<Point2f> points1;
		vector<Point2f> points2;
		vector<float> similarities;
		for(int k = 0; k < matches[i].size(); k++){
			points1.push_back(keypoints[i][matches[i][k].queryIdx].pt);
			points2.push_back(keypoints[i+1][matches[i][k].trainIdx].pt);
			similarities.push_back(matches[i][k].distance);
		}
		vector<Point2f> bestValidPoint1, bestValidPoint2;
		Mat mask = masks[i];
		for(int j = 0; j < mask.total(); j++){
			if(mask.at<uchar>(j)){
				bestValidPoint1.push_back(points1[j]);
				bestValidPoint2.push_back(points2[j]);
				break;
			}
		}

		Mat E, R, t;
		E = Estimator::essentialFromFundamental(fundMatrices[i], K1, K2);
		essenMatrices[i] = E;

		if(Estimator::motionFromEssential(E, K1, K2, bestValidPoint1, bestValidPoint2, R, t)){
			RtMatrices[i] = pair<Mat,Mat>(R,t);
			cout << R << endl;
			cout << t << endl;
		} else {
			cout << "Valid inter-camera motion could not be recovered" << endl;
		}

	}
}

void SFM::showCorrespondences(){
	for(int i = 0; i < matches.size(); i++){
		vector<DMatch> good_matches;
		for(int l = 0; l < matches[i].size(); l++){
			if(masks[i].at<uchar>(l)){
				good_matches.push_back(matches[i][l]);
			}
		}
		
		Mat img_matches;
		drawMatches( GRAY_imgs[i], keypoints[i], GRAY_imgs[i+1], keypoints[i+1],
	               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
	               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


		Size size(img_matches.cols/2, img_matches.rows/2);
		Mat dst;
		resize(img_matches,dst,size);
	  	// Show detected matches
	  	imshow( "Good Matches", dst );
	  	waitKey(-1);
	}
}
