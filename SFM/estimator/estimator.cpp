#include <stdio.h>
#include <string>
#include <vector>
#include <dirent.h>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include "estimator.hpp"

//RANSAC variations
#include "RANSAC.cpp"
#include "PROSAC.cpp"
#include "TddTest.cpp"
#include "PE-RANSAC.cpp"
#include "LO-RANSAC.cpp"

//COnstants
#include "../constants.hpp"

using namespace std;
using namespace cv;

int Estimator::num_iters = 0;
float Estimator::runtime = 0;
vector<int> Estimator::pool = vector<int>();

void Estimator::subselect(_InputArray _points1, _InputArray _points2, _OutputArray _output1, _OutputArray _output2, int len, int limit){
	Mat points1 = _points1.getMat(), points2 = _points2.getMat();
	int d1 = points1.channels() > 1 ? points1.channels() : points1.cols;
	bool multichan = points1.channels() > 1;
	int count = points1.checkVector(d1), count2 = points1.checkVector(d1);

	CV_Assert( points1.type() == points2.type() );
	CV_Assert( count == count2 );

	if(limit <= 0)
		limit = count;

	if(multichan){
		_output1.create(len, 1, points1.type());
		_output2.create(len, 1, points2.type());
	} else {
		_output1.create(len, 2, points1.type());
		_output2.create(len, 2, points2.type());
	}

	Mat output1 = _output1.getMat();
	Mat output2 = _output2.getMat();

	if(Estimator::pool.size() != count){
		Estimator::pool = vector<int>(count);
		for (int i=0; i<count; i++) Estimator::pool[i] = i;
	}

	random_shuffle(Estimator::pool.begin(), Estimator::pool.begin()+limit);
	
	for(int j=0; j<len; j++){
		points1.row(Estimator::pool[j]).copyTo(output1.row(j));
		points2.row(Estimator::pool[j]).copyTo(output2.row(j));
	}

}

//Directly ported from opencv ptsetreg.cpp
int Estimator::updateNumIters( double p, double ep, int modelPoints, int maxIters )
{

    p = MAX(p, 0.);
    p = MIN(p, 1.);
    ep = MAX(ep, 0.);
    ep = MIN(ep, 1.);

    // avoid inf's & nan's
    double num = max(1. - p, DBL_MIN);
    double denom = 1. - pow(1. - ep, modelPoints);
    if( denom < DBL_MIN )
        return 0;

    num = log(num);
    denom = log(denom);

    return denom >= 0 || -num >= maxIters*(-denom) ? maxIters : cvRound(num/denom);
}

//Ported from opencv
int Estimator::getInliers(_InputArray _points1, _InputArray _points2, Mat _F, double err, _OutputArray _mask){

	Mat __m1 = _points1.getMat(), __m2 = _points2.getMat();
    int i, count = __m1.checkVector(2);

    const Point2f* m1 = __m1.ptr<Point2f>();
    const Point2f* m2 = __m2.ptr<Point2f>();
    const double* F = _F.ptr<double>();

    uchar* mask;
    if(_mask.needed()){
    	_mask.create(count, 1, CV_8U);
    	mask = _mask.getMat().ptr<uchar>();
	} else {
		Mat tempMask;
		tempMask.create(count, 1, CV_8U);
		mask = tempMask.ptr<uchar>();
	}

    float thresh = (float)(err*err);
    int nInliers = 0;


    for( i = 0; i < count; i++ )
    {
        double a, b, c, d1, d2, s1, s2;

        a = F[0]*m1[i].x + F[1]*m1[i].y + F[2];
        b = F[3]*m1[i].x + F[4]*m1[i].y + F[5];
        c = F[6]*m1[i].x + F[7]*m1[i].y + F[8];

        s2 = 1./(a*a + b*b);
        d2 = m2[i].x*a + m2[i].y*b + c;

        a = F[0]*m2[i].x + F[3]*m2[i].y + F[6];
        b = F[1]*m2[i].x + F[4]*m2[i].y + F[7];
        c = F[2]*m2[i].x + F[5]*m2[i].y + F[8];

        s1 = 1./(a*a + b*b);
        d1 = m1[i].x*a + m1[i].y*b + c;

        if(mask[i] = ((float)max(d1*d1*s1, d2*d2*s2) <= thresh))
        	nInliers++;
    }

    return nInliers;
}

//Ported from github libmv, src/libmv/multiview/triangulation.cc
Mat Estimator::triangulateDLT(_InputArray _P1, _InputArray _P2, _InputArray _p1, _InputArray _p2){

	Mat trian(4, 4, CV_64F);
	Mat p1 = _p1.getMat(), p2 = _p2.getMat(), P1 = _P1.getMat(), P2 = _P2.getMat();

	CV_Assert(p1.dims == p2.dims);
	if(p1.dims > 1){
		p1.reshape(1,1);
		p2.reshape(1,1);
	}

	for (int i = 0; i < 4; i++) {
		trian.at<double>(0,i) = p1.at<float>(0) * P1.at<double>(2,i) - P1.at<double>(0,i);
		trian.at<double>(1,i) = p1.at<float>(1) * P1.at<double>(2,i) - P1.at<double>(1,i);
		trian.at<double>(2,i) = p2.at<float>(0) * P2.at<double>(2,i) - P2.at<double>(0,i);
		trian.at<double>(3,i) = p2.at<float>(1) * P2.at<double>(2,i) - P2.at<double>(1,i);
	}

	Mat d, U, Vt;
	SVD::compute(trian, d, U, Vt, SVD::FULL_UV);
	Mat homog_sol = Vt.t().col(trian.cols-1);

	Mat m(3, 1, CV_64F);
	double factor = homog_sol.at<double>(homog_sol.total()-1);
	for(int j = 0; j < 3; j++){
		m.at<double>(j) = homog_sol.at<double>(j)/factor;
	}
	return m;
}

//Ported from github libmv, src/libmv/multiview/fundamental.cc
bool Estimator::motionFromEssential(_InputArray _E, _InputArray _K1, _InputArray _K2, _InputArray _p1, _InputArray _p2, _OutputArray _R, _OutputArray _t){

	
	Mat K1 = _K1.getMat(), K2 = _K2.getMat();
	Mat U, d, Vt;
	SVD::compute(_E, d, U, Vt, SVD::FULL_UV);

	// Last column of U is undetermined since d = (a a 0).
	if (determinant(U) < 0) {
		U.col(2) *= -1;
	}

	// Last row of Vt is undetermined since d = (a a 0).
	if (determinant(Vt) < 0) {
		Vt.row(2) *= -1;
	}

	Mat W = (Mat_<double>(3,3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);

	Mat U_W_Vt = U * W * Vt;
	Mat U_Wt_Vt = U * W.t() * Vt;

	vector< pair<Mat,Mat> > motionMats;
	motionMats.push_back(pair<Mat,Mat>(U_W_Vt, U.col(2)));
	motionMats.push_back(pair<Mat,Mat>(U_W_Vt, -U.col(2)));
	motionMats.push_back(pair<Mat,Mat>(U_Wt_Vt, U.col(2)));
	motionMats.push_back(pair<Mat,Mat>(U_Wt_Vt, -U.col(2)));

	
	Mat P1 = Mat_<double>(3,4), P2 = Mat_<double>(3,4);
	Mat R1 = Mat::eye(3, 3, CV_64F);
	Mat t1 = Mat::zeros(3, 1, CV_64F);

	for(int h = 0; h < R1.cols; h++){
		R1.col(h).copyTo(P1.col(h));
	}
	t1.copyTo(P1.col(3));
	P1 = K1 * P1;
	
	for (int i = 0; i < motionMats.size(); i++) {
		Mat R2 = motionMats[i].first;
		Mat t2 = motionMats[i].second;
		
		for(int h = 0; h < R2.cols; h++){
			R2.col(h).copyTo(P2.col(h));
		}
		t2.copyTo(P2.col(3));
		P2 = K2 * P2;

		Mat X = Estimator::triangulateDLT(P1, P2, _p1, _p2);
		
		//Depths(z-coordinate)
		double d1 = Mat(R1*X).at<double>(2) + t1.at<double>(2);
		double d2 = Mat(R2*X).at<double>(2) + t2.at<double>(2);
		// Test if point is in front of the cameras
		if (d1 > 0 && d2 > 0) {
			R2.copyTo(_R);
			t2.copyTo(_t);
			return true;
		}
	}
	
	//Could not recover motion data
	return false;
}


//Ported from SFM opencv module
Mat Estimator::essentialFromFundamental(_InputArray _F, _InputArray _K1, _InputArray _K2){
	const Mat F = _F.getMat(), K1 = _K1.getMat(), K2 = _K2.getMat();
    const int depth =  F.depth();

    CV_Assert(F.cols == 3 && F.rows == 3);
    CV_Assert((depth == CV_32F || depth == CV_64F));

    Mat E(3, 3, depth);
    E = K2.t() * F * K1;

    return E;
}

//Directly ported from opencv
int Estimator::fundMat(_InputArray _points1, _InputArray _points2, _OutputArray _F, bool useAll = false){

	Mat _m1 = _points1.getMat(), _m2 = _points2.getMat();
	Point2d m1c(0,0), m2c(0,0);
    double t, scale1 = 0, scale2 = 0;

    const Point2f* m1 = _m1.ptr<Point2f>();
    const Point2f* m2 = _m2.ptr<Point2f>();
    CV_Assert( (_m1.cols == 1 || _m1.rows == 1) && _m1.size() == _m2.size());
    int i, count = _m1.checkVector(2);

    if(!useAll)
    	count = (count < MIN_MODEL_POINTS)? count : MIN_MODEL_POINTS;

    // compute centers and average distances for each of the two point sets
    for( i = 0; i < count; i++ )
    {
        m1c += Point2d(m1[i].x, m1[i].y);
        m2c += Point2d(m2[i].x, m2[i].y);
    }

    t = 1./count;
    m1c *= t;
    m2c *= t;

    for( i = 0; i < count; i++ )
    {
        scale1 += norm(Point2d(m1[i].x - m1c.x, m1[i].y - m1c.y));
        scale2 += norm(Point2d(m2[i].x - m2c.x, m2[i].y - m2c.y));
    }

    scale1 *= t;
    scale2 *= t;

    if( scale1 < FLT_EPSILON || scale2 < FLT_EPSILON )
        return 0;

    scale1 = sqrt(2.)/scale1;
    scale2 = sqrt(2.)/scale2;

    Matx<double, 9, 9> A;

    for( i = 0; i < count; i++ )
    {
        double x1 = (m1[i].x - m1c.x)*scale1;
        double y1 = (m1[i].y - m1c.y)*scale1;
        double x2 = (m2[i].x - m2c.x)*scale2;
        double y2 = (m2[i].y - m2c.y)*scale2;
        Vec<double, 9> r( x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1 );
        A += r*r.t();
    }

    Vec<double, 9> W;
    Matx<double, 9, 9> V;

    eigen(A, W, V);

    for( i = 0; i < 9; i++ )
    {
        if( fabs(W[i]) < DBL_EPSILON )
            break;
    }

    if( i < 8 )
        return 0;

    Matx33d F0( V.val + 9*8 );

    Vec3d w;
    Matx33d U;
    Matx33d Vt;

    SVD::compute( F0, w, U, Vt);
    w[2] = 0.;

    F0 = U*Matx33d::diag(w)*Vt;

    Matx33d T1( scale1, 0, -scale1*m1c.x, 0, scale1, -scale1*m1c.y, 0, 0, 1 );
    Matx33d T2( scale2, 0, -scale2*m2c.x, 0, scale2, -scale2*m2c.y, 0, 0, 1 );

    F0 = T2.t()*F0*T1;

    // make F(3,3) = 1
    if( fabs(F0(2,2)) > FLT_EPSILON )
        F0 *= 1./F0(2,2);


    Mat(F0).copyTo(_F);

    return 1;
}

void Estimator::debug(_InputArray _points1, _InputArray _points2, _OutputArray _F){
	Estimator::fundMat(_points1.getMat(), _points2.getMat(), _F);
}

Mat Estimator::estFundamentalMat(_InputArray _points1, _InputArray _points2,
                                int method, double param1 = 1, double param2 = 0.99, _OutputArray _mask = _OutputArray()){
	return Estimator::estFundamentalMat(_points1, _points2, method, param1, param2, INLIER_RATIO, _mask, vector<float>());
}
Mat Estimator::estFundamentalMat(_InputArray _points1, _InputArray _points2,
                                int method, double param1 = 1, double param2 = 0.99, _OutputArray _mask = _OutputArray(), vector<float> similarities = vector<float>()){
	return Estimator::estFundamentalMat(_points1, _points2, method, param1, param2, INLIER_RATIO, _mask, similarities);
}
Mat Estimator::estFundamentalMat(_InputArray _points1, _InputArray _points2,
                                int method, double param1 = 1, double param2 = 0.99, double param3 = INLIER_RATIO, _OutputArray _mask = _OutputArray()){
	return Estimator::estFundamentalMat(_points1, _points2, method, param1, param2, param3, _mask, vector<float>());
}

Mat Estimator::estFundamentalMat(_InputArray _points1, _InputArray _points2,
                                int method, double param1 = 1, double param2 = 0.99, double param3 = INLIER_RATIO, _OutputArray _mask = _OutputArray(), vector<float> similarities = vector<float>()){
		
	//Get Matrices
	Mat points1 = _points1.getMat(), points2 = _points2.getMat();
    Mat m1, m2, F, nm1, nm2, T1, T2;

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

    //cout << npoints << endl;
    _mask.create(npoints, 1, CV_8U, -1, true);
    Mat mask = _mask.getMat();
    CV_Assert( (mask.cols == 1 || mask.rows == 1) && (int)mask.total() == npoints );
    mask.setTo(Scalar::all(1));

    // Result, returns number of inliers
	int result = 0;

    FundMatEstimator* fme = createFundMatEstimator(method, param1, param2, param3);
    fme->run(m1, m2, F, _mask, similarities);
    result = countNonZero(mask);

    Mat filteredPoints1(result, m1.cols, m1.type()), filteredPoints2(result, m2.cols, m2.type());
   	int k = 0;
    for(int j = 0; j < mask.total() && k < result; j++){
    	if(mask.at<uchar>(j)){
    		m1.row(j).copyTo(filteredPoints1.row(k));
    		m2.row(j).copyTo(filteredPoints2.row(k));
    		k++;
		}
    }

    //Re-weighting, improving accuracy
    F = findFundamentalMat(filteredPoints1, filteredPoints2, CV_FM_LMEDS, param1, param2);

    Estimator::num_iters = fme->lastNumOfIterations();
    Estimator::runtime = fme->lastRuntime();

    //IN case if inliers are lower than required
    if( result < 8)
    	return Mat();

	return F;
}

FundMatEstimator* Estimator::createFundMatEstimator(int method, double param1, double param2, double param3 = INLIER_RATIO){
	switch(method) {
		case SFM_RANSAC :
			return (new RANSAC_Estimator(param1, param2, param3));
			break;
		case SFM_RANSAC_Tdd :
			return (new TddTest_Estimator(param1, param2, param3, false));
			break;
		case SFM_PROSAC_Tdd :
			return (new TddTest_Estimator(param1, param2, param3, true));
			break;
		case SFM_PE_RANSAC :
			return (new PE_RANSAC_Estimator(param1, param2, param3));
			break;
		case SFM_PROSAC :
			return (new PROSAC_Estimator(param1, param2, param3));
			break;
		case SFM_LO_RANSAC :
			return (new LO_RANSAC_Estimator(param1, param2, param3));
			break;
		default :
			return NULL;
	}
}

bool Estimator::triangulateViews(vector<Point2f> points1, vector<Point2f> points2, 
                                 Matx34f Pleft, Matx34f Pright, vector<intPair> indxs, intPair pr, Mat K, PointCloud_f& pointCloud){
	Mat normalizedLeftPts;
	Mat normalizedRightPts;
	undistortPoints(points1, normalizedLeftPts,  K, Mat());
	undistortPoints(points2, normalizedRightPts, K, Mat());

	Mat pHomogeneous;
	triangulatePoints(Pleft, Pright, normalizedLeftPts, normalizedRightPts, pHomogeneous);

	Mat points3f;
	convertPointsFromHomogeneous(pHomogeneous.t(), points3f);

	Mat rvecLeft;
    Rodrigues(Pleft.get_minor<3, 3>(0, 0), rvecLeft);
    Mat tvecLeft(Pleft.get_minor<3, 1>(0, 3).t());

    vector<Point2f> projectedOnLeft(points1.size());
    projectPoints(points3f, rvecLeft, tvecLeft, K, Mat(), projectedOnLeft);

    Mat rvecRight;
    Rodrigues(Pright.get_minor<3, 3>(0, 0), rvecRight);
    Mat tvecRight(Pright.get_minor<3, 1>(0, 3).t());

    vector<Point2f> projectedOnRight(points2.size());
    projectPoints(points3f, rvecRight, tvecRight, K, Mat(), projectedOnRight);

    for (int l = 0; l < points3f.rows; l++) {
        //check if point reprojection error is small enough
        if (cv::norm(projectedOnLeft[l]  - points1[l])  > MIN_REPR_ERROR ||
            cv::norm(projectedOnRight[l] - points2[l]) > MIN_REPR_ERROR)
        {
            continue;
        }

        Mapped3DPoint_f p;
        p.p = Point3f(points3f.at<float>(l, 0),
                      points3f.at<float>(l, 1),
                      points3f.at<float>(l, 2)
                      );

        //use back reference to point to original features in images
        p.views[pr.first]  = indxs[l].first;
        p.views[pr.second] = indxs[l].second;

        pointCloud.push_back(p);
    }

    return pointCloud.size() > 0;
}