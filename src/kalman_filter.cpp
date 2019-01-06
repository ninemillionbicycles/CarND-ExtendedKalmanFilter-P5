#include "kalman_filter.h"
#include "tools.h"
#include <math.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/* 
 * Please note that the Eigen library does not initialize 
 * VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
   * TODO: predict the state
   */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
   * TODO: update the state by using Kalman Filter equations
   */
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  // MatrixXd Si = S.inverse();
  // MatrixXd PHt = P_ * Ht;
  MatrixXd K = P_ * Ht * S.inverse();

  // new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * TODO: update the state by using Extended Kalman Filter equations
   */
  Tools tools;

  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  VectorXd hx;
  hx <<   0.0,
          0.0,
          0.0;

  if (px != 0) { // don't divide by zero
    hx << sqrt(px*px + py*py),
          atan2(py,px), // must be between -pi and pi
          (px*vx + py*vy) / sqrt(px*px + py*py);
  }

  VectorXd y = z - hx; 
  y(1) = fmod(y(1) + M_PI, 2*M_PI) - M_PI; // make sure y(1) is between -pi and pi

  MatrixXd Hj = tools.CalculateJacobian(x_);

  MatrixXd Ht = Hj.transpose();
  MatrixXd S = Hj * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();

  // new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
