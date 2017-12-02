#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

#define eps 0.000001

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
	// if this is false, laser measurements will be ignored (except during init)
	use_laser_ = true;

	// if this is false, radar measurements will be ignored (except during init)
	use_radar_ = true;

	// Process noise standard deviation longitudinal acceleration in m/s^2
	std_a_ = 2;

	// Process noise standard deviation yaw acceleration in rad/s^2
	std_yawdd_ = 0.5;

	// Laser measurement noise standard deviation position1 in m
	std_laspx_ = 0.15;

	// Laser measurement noise standard deviation position2 in m
	std_laspy_ = 0.15;

	// Radar measurement noise standard deviation radius in m
	std_radr_ = 0.3;

	// Radar measurement noise standard deviation angle in rad
	std_radphi_ = 0.03;

	// Radar measurement noise standard deviation radius change in m/s
	std_radrd_ = 0.3;

	//set state dimension
	n_x_ = 5;

	//set augmented dimension
	n_aug_ = 7;

	//define spreading parameter
	lambda_ = 3 - n_aug_;

	//define start time
	time_us_ = 0;

	// initial state vector
    x_ = VectorXd(5);

    // initial covariance matrix
    P_ = MatrixXd(5, 5);

    //create matrix with predicted sigma points as columns
    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

    //create vector for weights
    weights_ = VectorXd(2*n_aug_+1);

    //define initialization status
    is_initialized_ = false;

  /*
    ToDo: one or more values initialized above might be wildly off...
  */
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
	Tools tools;
	/*****************************************************************************
	*  Initialization
	****************************************************************************/
	if (!is_initialized_) {
		// initialize state vector
		x_ = VectorXd(5);
		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
	    	VectorXd z = VectorXd(3);
	    	z = meas_package.raw_measurements_;
	    	x_ << z[0]*sin(z[1]), z[0]*cos(z[1]), 0, 0, 0;
	    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
	    	x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
	    } else {
	    	x_ << 0, 0, 0, 0, 0;
	    }

		// initialize state covariance matrix
		P_ << 0.1, 0, 0, 0, 0,
		      0, 0.1, 0, 0, 0,
		      0, 0, 10, 0, 0,
		      0, 0, 0, 1, 0,
			  0, 0, 0, 0, 1;

		// set weights (code from UKF tutorial)
		double weight_0 = lambda_/(lambda_+n_aug_);
		weights_(0) = weight_0;
		for(int i=1; i<2*n_aug_+1; i++) {  //2n+1 weights
			double weight = 0.5/(n_aug_+lambda_);
			weights_(i) = weight;
		}

	    time_us_ = meas_package.timestamp_;
	    // done initializing, no need to predict or update
	    is_initialized_ = true;
	    return;
	}

	/*****************************************************************************
	*  Prediction
	****************************************************************************/

	float dt = (meas_package.timestamp_ - time_us_)/1000000.;
	time_us_ = meas_package.timestamp_;

	Prediction(dt);

	/*****************************************************************************
	 *  Update
	 ****************************************************************************/

	if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_ == true) {
		UpdateRadar(meas_package);
	} else if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_ == true) {
		UpdateLidar(meas_package);
	}

	// print the output
	cout << "x_ = " << x_ << endl;
	cout << "P_ = " << P_ << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
	//create augmented mean vector
	VectorXd x_aug = VectorXd(n_aug_);
	//create augmented mean state
	x_aug << x_,0,0;

	//create augmented state covariance
	MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
	//create augmented covariance matrix
	P_aug.fill(0.0);
	P_aug.topLeftCorner(5,5) = P_;
	P_aug(5,5) = std_a_*std_a_;
	P_aug(6,6) = std_yawdd_*std_yawdd_;

	//create sigma point matrix
	MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

	//create square root matrix
	MatrixXd L = P_aug.llt().matrixL();

	//create augmented sigma points
	Xsig_aug.col(0)  = x_aug;
	for (int i = 0; i< n_aug_; i++) {
		Xsig_aug.col(i+1)        = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
		Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
	}

	float delta_t2 = delta_t*delta_t;
	//predict sigma points
	for(int i=0; i<2*n_aug_+1; i++) {
		float px       = Xsig_aug(0,i);
		float py       = Xsig_aug(1,i);
		float v        = Xsig_aug(2,i);
		float yaw      = Xsig_aug(3,i);
		float yawd     = Xsig_aug(4,i);
		float nu_a     = Xsig_aug(5,i);
		float nu_yawdd = Xsig_aug(6,i);

		if(fabs(yawd)<eps) {
			Xsig_pred_.col(i) << px+v*cos(yaw)*delta_t+0.5*delta_t2*cos(yaw)*nu_a,
								 py+v*sin(yaw)*delta_t+0.5*delta_t2*sin(yaw)*nu_a,
								 v+delta_t*nu_a,
								 yaw+0.5*delta_t2*nu_yawdd,
								 yawd+delta_t*nu_yawdd;
		} else {
			Xsig_pred_.col(i) << px+v/yawd*(sin(yaw+yawd*delta_t)-sin(yaw))+0.5*delta_t2*cos(yaw)*nu_a,
								 py+v/yawd*(-cos(yaw+yawd*delta_t)+cos(yaw))+0.5*delta_t2*sin(yaw)*nu_a,
								 v+delta_t*nu_a,
								 yaw+yawd*delta_t+0.5*delta_t2*nu_yawdd,
								 yawd+delta_t*nu_yawdd;
		}
	}

	//predicted state mean (code from UKF tutorial)
	x_.fill(0.0);
	for(int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
		x_ = x_ + weights_(i) * Xsig_pred_.col(i);
	}

	//predicted state covariance matrix (code from UKF tutorial)
	P_.fill(0.0);
	for(int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
		// state difference
	    VectorXd x_diff = Xsig_pred_.col(i) - x_;
	    //angle normalization
	    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
	    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

	    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
	}
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
	/**
	TODO:
	You'll also need to calculate the lidar NIS.
	*/

	//set measurement dimension, lidar can measure px and py
	int n_z = 2;

	//create matrix for sigma points in measurement space (code from UKF tutorial)
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

	//mean predicted measurement (code from UKF tutorial)
	VectorXd z_pred = VectorXd(n_z);

	//measurement covariance matrix S (code from UKF tutorial)
	MatrixXd S = MatrixXd(n_z,n_z);

	//create matrix for cross correlation Tc (code from UKF tutorial)
	MatrixXd Tc = MatrixXd(n_x_, n_z);

	//transform sigma points into measurement space
	for(int i=0; i<2*n_aug_+1; i++) {
			float px   = Xsig_pred_(0,i);
		    float py   = Xsig_pred_(1,i);
		    Zsig.col(i) << px, py;
	}
	//calculate mean predicted measurement (code from UKF tutorial)
	z_pred.fill(0.0);
	for(int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}
	//calculate measurement covariance matrix S and cross correlation matrix Tc
	S.fill(0.0);
	Tc.fill(0.0);
	for(int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		//angle normalization
		while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
		while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

	    //measurement difference
		VectorXd z_diff = Zsig.col(i) - z_pred;

	    S = S + weights_(i) * z_diff * z_diff.transpose();
	    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}
	MatrixXd R = MatrixXd(n_z,n_z);
	R << pow(std_laspx_,2), 0.,
		 0., pow(std_laspy_,2);
	S = S + R;

	//calculate Kalman gain K;
	MatrixXd K = Tc*S.inverse();
	//update state mean and covariance matrix
	VectorXd innovation = meas_package.raw_measurements_ - z_pred;
	x_ = x_ + K*innovation;
	P_ = P_ - K*S*K.transpose();

	NIS_lidar_ = innovation.transpose() * S.inverse() * innovation;
	cout << "NIS-Lidar = " << NIS_lidar_ << endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
	/**
	TODO:
	You'll also need to calculate the radar NIS.
	*/

	//set measurement dimension, radar can measure r, phi, and r_dot (code from UKF tutorial)
	int n_z = 3;

	//create matrix for sigma points in measurement space (code from UKF tutorial)
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

	//mean predicted measurement (code from UKF tutorial)
	VectorXd z_pred = VectorXd(n_z);

	//measurement covariance matrix S (code from UKF tutorial)
	MatrixXd S = MatrixXd(n_z,n_z);

	//create matrix for cross correlation Tc (code from UKF tutorial)
	MatrixXd Tc = MatrixXd(n_x_, n_z);

	//transform sigma points into measurement space
	for(int i=0; i<2*n_aug_+1; i++) {
		float px   = Xsig_pred_(0,i);
	    float py   = Xsig_pred_(1,i);
	    float v    = Xsig_pred_(2,i);
	    float yaw  = Xsig_pred_(3,i);
	    float yawd = Xsig_pred_(4,i);
	    Zsig.col(i) << sqrt(pow(px,2)+pow(py,2)),
	    			   atan2(py,px),
					   (px*cos(yaw)*v+py*sin(yaw)*v)/(sqrt(pow(px,2)+pow(py,2)));
	}
	//calculate mean predicted measurement (code from UKF tutorial)
	z_pred.fill(0.0);
	for(int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}
	//calculate measurement covariance matrix S and cross correlation matrix Tc
	S.fill(0.0);
	Tc.fill(0.0);
	for(int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		VectorXd z_diff = Zsig.col(i) - z_pred;
	    //angle normalization
		while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
		while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
	    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
	    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

	    S = S + weights_(i) * z_diff * z_diff.transpose();
	    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}
	MatrixXd R = MatrixXd(n_z,n_z);
	R << pow(std_radr_,2), 0., 0.,
		 0., pow(std_radphi_,2), 0.,
		 0., 0., pow(std_radrd_,2);
	S = S + R;

	//calculate Kalman gain K;
	MatrixXd K = Tc*S.inverse();
	//update state mean and covariance matrix
	VectorXd innovation = meas_package.raw_measurements_ - z_pred;
	x_ = x_ + K*innovation;
	P_ = P_ - K*S*K.transpose();

	NIS_radar_ = innovation.transpose() * S.inverse() * innovation;
	cout << "NIS-Radar = " << NIS_radar_ << endl;
}
