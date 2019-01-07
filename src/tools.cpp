#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	/**
	* Calculate the RMSE.
	*/
	VectorXd rmse = VectorXd(4);
	rmse << 0,0,0,0;

	// check the validity of the inputs:
	//  * the estimation vector size should not be zero
	//  * the estimation vector size should equal ground truth vector size
	if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
		std::cout << "Invalid estimation or ground_truth data" << std::endl;
		return rmse;
	}

	// accumulate squared residuals
	for (unsigned int i = 0; i < estimations.size(); ++i) {

		VectorXd residual = estimations[i] - ground_truth[i];

		// coefficient-wise multiplication
		// residual = residual.array()*residual.array();
		residual = residual.array().square();
		rmse += residual;
	}

	// calculate the mean
	rmse = rmse/estimations.size();

	// calculate the squared root
	rmse = rmse.array().sqrt();

	// return the result
	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * Calculate a Jacobian.
   */
  MatrixXd Hj(3,4);
  
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  Hj <<	0.0, 0.0, 0.0, 0.0,
  			0.0, 0.0, 0.0, 0.0,
  			0.0, 0.0, 0.0, 0.0;

  // check division by zero
  if(px == 0 & py == 0) {
      std::cout<<"Error: Division by 0!";
  }
  
  // compute the Jacobian matrix
  else {
      float px2 = px*px;
      float py2 = py*py;
      float pxy2 = px2 + py2;
      Hj << px/sqrt(pxy2), py/sqrt(pxy2), 0.0, 0.0,
            -py/pxy2, px/pxy2, 0.0, 0.0,
            py*(vx*py-vy*px)/pow(pxy2,1.5), px*(vy*px-vx*py)/pow(pxy2,1.5), px/sqrt(pxy2), py/sqrt(pxy2);
  }

  return Hj;
}
