/*
 * Arrows.hpp
 *
 *  Created on: 3 Dec 2011
 *      Author: ejm
 */

#ifndef ARROWS_HPP
#define ARROWS_HPP

#include <iostream>
#include <math.h>
#define _USE_MATH_DEFINES

#include "utils.hpp"
#include "Box.hpp"

using blitz::Array;

class Arrows {
public:
	Arrows(
			const Box& box, int num_arrows,
			float sense_grad,
			float t_mem, float sense_mem,
			bool onsided_flag, char rate_alg, char bc_alg);
	void rs_update(const Box& box);
	void vs_update(void);
	void rates_update(const Box& box);
	const Array<float, DIM>& rs_get(void) const;
	const Array<float, DIM>& vs_get(void) const;
	int num_arrows_get(void) const;

private:
	void rs_initialise(const Box& box);
	void vs_initialise();
	void rates_initialise(char rate_algf, float sense_mem, float t_mem);
	void tumble(void);
	float grad_find(const Box& box, Array<float, DIM>& field, int i_arrow);
	float integral_find(int i_arrow);
	void rates_update_const(const Box& box);
	void rates_update_grad(const Box& box);
	void rates_update_mem(const Box& box);
	void wall_handle(int i_arrow, Range i_dims_hit);
	void wall_specular(int i_arrow, Range i_dims_hit);
	void wall_bounceback(int i_arrow, Range i_dims_hit);
	void wall_aligning(int i_arrow, Range i_dims_hit);
	void wall_stalling(int i_arrow, Range i_dims_hit);

	int m_num_arrows;
	float m_v;
	void (Arrows::*m_prates_update)(const Box&);
	void (Arrows::*m_pwall_handle)(int i_arrow, Range i_dims_hit);

	Array<float, 2> m_rs, m_vs;

	Array<float, 1> m_K, m_rates;
	Array<float, 2> m_attract_mem;
	float m_rate_base, m_sense_grad;

	char m_rate_alg, m_bc_alg;
	bool m_onesided_flag;
};

#endif /* ARROWS_HPP */
