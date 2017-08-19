#include "ba.hpp"
#include "../constants.hpp"

using namespace std;
using namespace cv;
using namespace ceres;

BundleAdjustment::BundleAdjustment(){
    initLogging();
}

//Enable google logging
bool BundleAdjustment::logging = false;
void BundleAdjustment::initLogging(){
    if(!logging){
        google::InitGoogleLogging("SFM");
        logging = true;
    }
} 

ceres::CostFunction* BundleAdjustment::createCostFunction(const double observed_x, const double observed_y, const int method){
    switch(method){
        case BA_RE_SIMPLE:
        return (new ceres::AutoDiffCostFunction<SimpleReprojectionError, 2, 6, 3, 1>(
                new SimpleReprojectionError(observed_x, observed_y)));
        break;
        case BA_RE_SNAVELY:
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
                new SnavelyReprojectionError(observed_x, observed_y)));
        break;
    };
}

bool BundleAdjustment::run(PointCloud_d& pointCloud, vector<Matx34d>& cameraPoses, _InputArray _K, const vector<Features_d>& image2dFeatures, int method, _OutputArray _Kopt){
    ceres::Problem problem;
    Mat K = _K.getMat();
    if(_Kopt.needed())
        K.copyTo(_Kopt);

    //Rtf, Rotation(angle-axis)[3], Translation[3], focal[1]
    vector< Matx<double, 1, 6> > cameraPoses6d(cameraPoses.size());
    cameraPoses6d.reserve(cameraPoses.size());
    for (size_t i = 0; i < cameraPoses.size(); i++) {
        const Matx34d& pose = cameraPoses[i];

        if (pose(0, 0) == 0 and pose(1, 1) == 0 and pose(2, 2) == 0) {
            //This camera pose is empty, it should not be used in the optimization
            cameraPoses6d.push_back(Matx<double, 1, 6>());
            continue;
        }
        Vec3d t(pose(0, 3), pose(1, 3), pose(2, 3));
        Matx33d R = pose.get_minor<3, 3>(0, 0);
        double angleAxis[3];
        ceres::RotationMatrixToAngleAxis<double>(R.t().val, angleAxis); //Ceres assumes col-major...

        cameraPoses6d.push_back(Matx<double, 1, 6>(
                angleAxis[0],
                angleAxis[1],
                angleAxis[2],
                t(0),
                t(1),
                t(2)));
    }

    //focal-length factor for optimization
    double focal = K.at<double>(0, 0);

    vector<Vec3d> points3d(pointCloud.size());

    for (int i = 0; i < pointCloud.size(); i++) {
        const Mapped3DPoint_d& p = pointCloud[i];
        points3d[i] = Vec3d(p.p.x, p.p.y, p.p.z);

        for (map<int,int>::const_iterator it = p.views.begin(); it != p.views.end(); it++) {
            //it->first  = camera index
            //it->second = 2d feature index
            Point2f p2f = image2dFeatures[it->first].points[it->second];

            //subtract center of projection, since the optimizer doesn't know what it is
            p2f.x -= K.at<double>(0, 2);
            p2f.y -= K.at<double>(1, 2);

            // Each Residual block takes a point and a camera as input and outputs a 2
            // dimensional residual. Internally, the cost function stores the observed
            // image location and compares the reprojection against the observation.
            ceres::CostFunction* cost_function = createCostFunction(p2f.x, p2f.y, method);

            problem.AddResidualBlock(cost_function,
                    NULL, /* squared loss */
                    cameraPoses6d[it->first].val,
                    points3d[i].val,
                    &focal);
        }
    }

    //Make Ceres automatically detect the bundle structure
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 500;
    options.eta = 1e-2;
    options.max_solver_time_in_seconds = 10;
    ceres::LoggingType ltypeInstance;
    ltypeInstance = SILENT;
    options.logging_type = ltypeInstance;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";

    //Ceres failed to converge
    if (!(summary.termination_type == ceres::CONVERGENCE)) {
        cerr << "Bundle adjustment failed." << endl;
        return false;
    }

    //Update optimized focal
    if(_Kopt.needed()){
        Mat Kopt = _Kopt.getMat();
        Kopt.at<double>(0, 0) = focal;
        Kopt.at<double>(1, 1) = focal;
    }

    //Implement the optimized camera poses and 3D points back into the reconstruction
    for (int i = 0; i < cameraPoses.size(); i++) {
        Matx34d& pose = cameraPoses[i];
        Matx34d poseBefore = pose;

        if (pose(0, 0) == 0 and pose(1, 1) == 0 and pose(2, 2) == 0) {
            //This camera pose is empty, it was not used in the optimization
            continue;
        }

        //Convert optimized Angle-Axis back to rotation matrix
        double rotationMat[9] = { 0 };
        ceres::AngleAxisToRotationMatrix(cameraPoses6d[i].val, rotationMat);

        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                pose(c, r) = rotationMat[r * 3 + c]; //`rotationMat` is col-major...
            }
        }

        //Translation
        pose(0, 3) = cameraPoses6d[i](3);
        pose(1, 3) = cameraPoses6d[i](4);
        pose(2, 3) = cameraPoses6d[i](5);
    }

    for (int i = 0; i < pointCloud.size(); i++) {
        pointCloud[i].p.x = points3d[i](0);
        pointCloud[i].p.y = points3d[i](1);
        pointCloud[i].p.z = points3d[i](2);
    }

    return true;
}
