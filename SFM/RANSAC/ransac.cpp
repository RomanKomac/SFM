#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <dirent.h>
#include <fstream>
#include "ransac.hpp"

using namespace std;
using namespace cv;


Mat Estimator::estFundamentalMat( _InputArray _points1, _InputArray _points2,
                                int method, double param1 = 1, double param2 = 0.99, _OutputArray _mask = _OutputArray(), vector<float> similarities = vector<float>()){
		
	//Get Matrices
	Mat points1 = _points1.getMat(), points2 = _points2.getMat();
    Mat m1, m2, F;

    //Number of points, determines if algorithm could be run
    int npoints = -1;

    for( int i = 1; i <= 2; i++ )
    {
        Mat& p = i == 1 ? points1 : points2;
        Mat& m = i == 1 ? m1 : m2;
        npoints = p.checkVector(2, -1, false);
        if( npoints < 0 )
        {
            npoints = p.checkVector(3, -1, false);
            //if( npoints < 0 )
                //CV_Error(Error::StsBadArg, "The input arrays should be 2D point sets");
            if( npoints == 0 )
                return Mat();
            convertPointsFromHomogeneous(p, p);
        }
        p.reshape(2, npoints).convertTo(m, CV_32F);
    }

    CV_Assert( m1.checkVector(2) == m2.checkVector(2) );

    /*
    if( npoints < 7 )
        return Mat();

    Ptr<PointSetRegistrator::Callback> cb = makePtr<FMEstimatorCallback>();
    int result;

    if( npoints == 7 || method == FM_8POINT )
    {
        result = cb->runKernel(m1, m2, F);
        if( _mask.needed() )
        {
            _mask.create(npoints, 1, CV_8U, -1, true);
            Mat mask = _mask.getMat();
            CV_Assert( (mask.cols == 1 || mask.rows == 1) && (int)mask.total() == npoints );
            mask.setTo(Scalar::all(1));
        }
    }
    else
    {
        if( param1 <= 0 )
            param1 = 3;
        if( param2 < DBL_EPSILON || param2 > 1 - DBL_EPSILON )
            param2 = 0.99;

        if( (method & ~3) == FM_RANSAC && npoints >= 15 )
            result = createRANSACPointSetRegistrator(cb, 7, param1, param2)->run(m1, m2, F, _mask);
        else
            result = createLMeDSPointSetRegistrator(cb, 7, param2)->run(m1, m2, F, _mask);
    }
	
	*/


    if( _mask.needed() )
    {
        _mask.create(npoints, 1, CV_8U, -1, true);
        Mat mask = _mask.getMat();
        CV_Assert( (mask.cols == 1 || mask.rows == 1) && (int)mask.total() == npoints );
        mask.setTo(Scalar::all(1));
	}

    // Result, returns 1 if successful
	int result;

    result = createFundMatEstimator(method, param1, param2, _mask, similarities)->run(m1, m2, F);

    if( result <= 0)
    	return Mat();

	return F;
}


class RANSAC_Estimator : public FundMatEstimator{
	public:
		RANSAC_Estimator(double param1, double param2){

		}
		int run(cv::Mat m1, cv::Mat m2, cv::Mat F){

			return 0;
		}
};
class PE_RANSAC_Estimator : public FundMatEstimator{
	public:
		PE_RANSAC_Estimator(double param1, double param2){

		}
		int run(cv::Mat m1, cv::Mat m2, cv::Mat F){

			return 0;
		}
};
class PROSAC_Estimator : public FundMatEstimator{
	public:
		PROSAC_Estimator(double param1, double param2){

		}
		int run(cv::Mat m1, cv::Mat m2, cv::Mat F){

			return 0;
		}
};
class MLESAC_Estimator : public FundMatEstimator{
	public:
		MLESAC_Estimator(double param1, double param2){

		}
		int run(cv::Mat m1, cv::Mat m2, cv::Mat F){

			return 0;
		}
};
class ARRSAC_Estimator : public FundMatEstimator{
	public:
		ARRSAC_Estimator(double param1, double param2){

		}
		int run(cv::Mat m1, cv::Mat m2, cv::Mat F){

			return 0;
		}
};

FundMatEstimator* Estimator::createFundMatEstimator(int method, double param1, double param2, _OutputArray _mask, vector<float> similarities){
	switch(method) {
		case SFM_RANSAC :
			return (new RANSAC_Estimator(param1, param2));
			break;
		case SFM_PE_RANSAC :
			return (new PE_RANSAC_Estimator(param1, param2));
			break;
		case SFM_PROSAC :
			return (new PROSAC_Estimator(param1, param2));
			break;
		case SFM_MLESAC :
			return (new MLESAC_Estimator(param1, param2));
			break;
		case SFM_ARRSAC :
			return (new ARRSAC_Estimator(param1, param2));
			break;
		default :
			return NULL;
	}
}