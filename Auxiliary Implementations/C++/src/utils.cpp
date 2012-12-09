/*
 * utils.cpp
 *
 *  Created on: 4 Dec 2011
 *      Author: ejm
 */

#include "utils.hpp"

boost::mt19937 gen(std::time(0));

float random_float(float min, float max) {
	boost::uniform_real<float> u(min, max);
	return u(gen);
}

int argmax(TinyVector<int, DIM>& v) {
	int i_max;
	int v_max;
	for (int i_dim = 0; i_dim < DIM; i_dim++)
		if ((i_dim == 0) or (v(i_dim) > v_max)) {
			i_max = i_dim;
			v_max = v(i_dim);
		}
	return i_max;
}

int argmax(Array<float, 1>& v) {
	int i_max;
	float v_max;
	for (int i_dim = 0; i_dim < v.extent(firstDim); i_dim++)
		if ((i_dim == 0) or (v(i_dim) > v_max)) {
			i_max = i_dim;
			v_max = v(i_dim);
		}
	return i_max;
}

Array<float, 1> polar_to_cart(Array<float, 1>& v_p) {
	Array<float, 1> v_c(v_p.shape());
	v_c(0) = v_p(0) * cos(v_p(1));
	v_c(1) = v_p(0) * sin(v_p(1));
	return v_c;
}

Array<float, 1> cart_to_polar(Array<float, 1>& v_c) {
	Array<float, 1> v_p(v_c.shape());
	v_p(0) = vector_mag(v_c);
	v_p(1) = atan2(v_c(1), v_c(0));
	return v_p;
}

float vector_mag(Array<float, 1>& v) {
	return sqrt(sum(pow2(v)));
}
