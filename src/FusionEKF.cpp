#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"
#include <math.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  // measurement covariance matrix - laser
  R_laser_ << 0.0225, 0.0,
              0.0, 0.0225;

  // measurement covariance matrix - radar
  R_radar_ << 0.09, 0.0, 0.0,
              0.0, 0.0009, 0.0,
              0.0, 0.0, 0.09;

  H_laser_ <<   1, 0, 0, 0,
                0, 1, 0, 0;

  // the initial transition matrix F_
  F_ = MatrixXd(4, 4);
  F_ << 1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1;

  // state covariance matrix P
  P_ = MatrixXd(4, 4);
  P_ << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1000, 0,
        0, 0, 0, 1000;

  Q_ = MatrixXd(4, 4);
  Q_ << 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */

  if (!is_initialized_) {
    VectorXd x_ = VectorXd(4);
    x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // set the state with the initial location and zero velocity
      x_ << measurement_pack.raw_measurements_[0]*cos(measurement_pack.raw_measurements_[1]), 
            measurement_pack.raw_measurements_[0]*sin(measurement_pack.raw_measurements_[1]), 
            0.0, 
            0.0;
      ekf_.Init(x_, P_, F_, Hj_, R_radar_, Q_);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // set the state with the initial location and zero velocity
      x_ <<   measurement_pack.raw_measurements_[0], 
              measurement_pack.raw_measurements_[1], 
              0.0, 
              0.0;
      ekf_.Init(x_, P_, F_, H_laser_, R_laser_, Q_);
    }

    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;

    return;
  }

  /**
   * Prediction
   */
  float noise_ax = 9.0;
  float noise_ay = 9.0;

  // compute the time elapsed between the current and previous measurements
  // dt - expressed in seconds
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;
  
  // modify the F matrix so that the time is integrated
  ekf_.F_(0,2) = dt;
  ekf_.F_(1,3) = dt;

  // set the process covariance matrix Q
  float dt2 = dt*dt;
  float dt3 = dt2*dt;
  float dt4 = dt3*dt;

  ekf_.Q_(0,0) = dt4/4*noise_ax;
  ekf_.Q_(1,1) = dt4/4*noise_ay;
  ekf_.Q_(0,2) = dt3/2*noise_ax;
  ekf_.Q_(2,0) = dt3/2*noise_ax;
  ekf_.Q_(1,3) = dt3/2*noise_ay;
  ekf_.Q_(3,1) = dt3/2*noise_ay;
  ekf_.Q_(2,2) = dt2*noise_ax;
  ekf_.Q_(3,3) = dt2*noise_ay;

  ekf_.Predict();

  /**
   * Update
   */
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // radar updates
    Hj_ = tools.CalculateJacobian(ekf_.x_);

    ekf_.Init(ekf_.x_, ekf_.P_, F_, Hj_, R_radar_, Q_);
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } 
  else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    // laser updates
    ekf_.Init(ekf_.x_, ekf_.P_, F_, H_laser_, R_laser_, Q_);
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  std::cout << "x_ = " << ekf_.x_ << endl;
  std::cout << "P_ = " << ekf_.P_ << endl;
}
