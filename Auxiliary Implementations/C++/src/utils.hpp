/*
 * utils.hpp
 *
 *  Created on: 4 Dec 2011
 *      Author: ejm
 */

#ifndef UTILS_HPP
#define UTILS_HPP

#include <math.h>
#include <ctime>

#include "blitz/array.h"
#include "blitz/tinyvec-et.h"
#include "boost/random.hpp"

using namespace blitz;

const float ZERO_THRESH = 1.0e-8;
const int DIM = 2;
const float DELTA_t = 0.001;
const firstIndex i1;

float random_float(float min, float max);
int argmax(TinyVector<int, DIM>& v);
int argmax(Array<float, 1>& v);
Array<float, 1> polar_to_cart(Array<float, 1>& v_p);
Array<float, 1> cart_to_polar(Array<float, 1>& v_c);
float vector_mag(Array<float, 1>& v);

#endif // UTILS_HPP
