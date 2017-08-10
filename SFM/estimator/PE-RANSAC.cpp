#include <stdio.h>
#include <string>
#include <vector>
#include <dirent.h>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include "estimator.hpp"
#include "../constants.hpp"

using namespace std;
using namespace cv;

class PE_RANSAC_Estimator : public FundMatEstimator{
	private:
		static bool sortPairs(pair< Mat,int > p1, pair< Mat,int > p2){
			return p1.second > p2.second;
		}
	public:
		PE_RANSAC_Estimator(double param1, double param2, double param3){
			reprojectError = param1;
			confidence = param2;
			inlier_ratio = param3;
			//Recommended values
			//M = number of generated hypotheses
			//B = number of points evaluated each time
			M = 500;
			B = 100;
		}
		int run(_InputArray _points1, _InputArray _points2, _OutputArray _F, _OutputArray _mask = _OutputArray(), vector<float> similarities = vector<float>()){

			//Reset number of iterations and runtime
			num_iters = 0;
			loop_runtime = 0;

			Mat points1 = _points1.getMat(), points2 = _points2.getMat();
			Mat mask, model, bestModel, subselect1, subselect2, bestMask1, bestMask2, subtest1, subtest2;

	        //Number of iterations set according to the standard termination criterion
	        int iter, niters = (int)ceil(log(1 - confidence)/log(1 - pow(inlier_ratio,8)));
	        int d1 = points1.channels() > 1 ? points1.channels() : points1.cols;
	        int d2 = points2.channels() > 1 ? points2.channels() : points2.cols;
	        int count = points1.checkVector(d1), count2 = points2.checkVector(d2), maxGoodCount = 0;

	        CV_Assert( confidence > 0 && confidence < 1 );
	        CV_Assert( count >= 0 && count2 == count );
	        if( count < MIN_MODEL_POINTS )
	            return 0;

	        B = (count < B)? count : B;

	        if( count == MIN_MODEL_POINTS )
	        {
	            Estimator::fundMat(points1, points2, bestModel, true);
	            bestModel.copyTo(_F);
	            bestMask1.setTo(Scalar::all(1));
	            return MIN_MODEL_POINTS;
	        }

	        clock_t t1,t2;
		    t1=clock();

			//Number of hypotheses yet to evaluate
			int nhypotheses = M;
			//Generate M initial hypotheses
			vector< pair< Mat,int > > hypotheses(nhypotheses);
			for(int j = 0; j < nhypotheses; j++){
				Estimator::subselect(points1, points2, subselect1, subselect2, MIN_MODEL_POINTS);
				if(Estimator::fundMat(subselect1, subselect2, model, true) > 0){
					hypotheses[j] = pair< Mat,int >(model.clone(),0);
				} else {
					j--;
				}
			}
			

			int i = 0;
			while(nhypotheses > 1){
				Estimator::subselect(points1, points2, subtest1, subtest2, B);
			 	for(int j = 0; j < nhypotheses; j++){
			 		num_iters++;
			 		hypotheses[j].second = Estimator::getInliers(subtest1, subtest2, hypotheses[j].first, reprojectError);
			 	}

			 	std::sort(hypotheses.begin(), hypotheses.begin()+nhypotheses, sortPairs);

			 	i++;
				nhypotheses = (int)floor(M * pow(2.,-i));
			}

			t2=clock();
		    loop_runtime = ((float)t2-(float)t1)/CLOCKS_PER_SEC;

		    hypotheses[0].first.copyTo(_F);

		    int finalCount = 0;
		    if(_mask.needed())
				finalCount = Estimator::getInliers(points1, points2, hypotheses[0].first, reprojectError, _mask);
			else
				finalCount = Estimator::getInliers(points1, points2, hypotheses[0].first, reprojectError);

			return finalCount;
		}
};