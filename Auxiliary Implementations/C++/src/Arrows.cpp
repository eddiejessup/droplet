/*
 * Arrows.cpp
 *
 *  Created on: 3 Dec 2011
 *      Author: ejm
 */

#include "Arrows.hpp"

Arrows::Arrows(const Box& box, int num_arrows,
							 float sense_grad,
							 float t_mem, float sense_mem,
							 bool onesided_flag, char rate_alg, char bc_alg) {

	m_num_arrows = num_arrows;
	m_onesided_flag = onesided_flag;
	m_rate_alg = rate_alg;
	m_bc_alg = bc_alg;

	float run_length_base = 1.0;
	float run_time_base = 1.0;

	m_rate_base = 1.0 / run_time_base;
	m_v = run_length_base * m_rate_base;

	if (bc_alg == 'p') {
		m_pwall_handle = &Arrows::wall_specular;
	} else if (bc_alg == 'a') {
		m_pwall_handle = &Arrows::wall_aligning;
	} else if (bc_alg == 't') {
		m_pwall_handle = &Arrows::wall_stalling;
	} else if (bc_alg == 'b') {
		m_pwall_handle = &Arrows::wall_bounceback;
	}

	if (rate_alg == 'c') {
		m_prates_update = &Arrows::rates_update_const;
	} else if (rate_alg == 'g') {
		m_prates_update = &Arrows::rates_update_grad;
		m_sense_grad = sense_grad;
	} else if (rate_alg == 'm') {
		m_prates_update = &Arrows::rates_update_mem;
	} else {
		cout << "Error: Rates update mode not recognised." << endl;
	}

	rs_initialise(box);
	vs_initialise();
	rates_initialise(rate_alg, sense_mem, t_mem);
}

void Arrows::rs_update(const Box& box) {
	Array<float, 1> test_r(DIM), test_cell_r(DIM), source_r_rel(DIM), v_a(DIM), sides(DIM);
	TinyVector<int, DIM> source_i, test_i, delta_i, pos_i;
	Range i_dims_hit;

	for(int i_arrow = 0; i_arrow < m_rs.extent(firstDim); i_arrow++) {
		test_r = m_rs(i_arrow, Range::all()) + m_vs(i_arrow, Range::all()) * DELTA_t;
		test_i = box.r_to_i(test_r);
		if(box.is_wall(test_i)) {
			test_cell_r = box.i_to_r(test_i);
			source_i = box.r_to_i(m_rs(i_arrow, Range::all()));
			source_r_rel = m_rs(i_arrow, Range::all()) - test_cell_r;
			for(int i_dim = 0; i_dim < DIM; i_dim++)
				sides(i_dim) = copysign(1.0, source_r_rel(i_dim));
			delta_i = abs(test_i - source_i);
			int adjacentness = sum(delta_i);
			if(adjacentness == 1) {
				int i_dim_hit = argmax(delta_i);
				i_dims_hit = Range(i_dim_hit, i_dim_hit);
			} else if(adjacentness == 2) {
				v_a = abs(m_vs(i_arrow, Range::all()));
				int i_dim_bias = argmax(v_a);
				pos_i = test_i;
				pos_i(1 - i_dim_bias) += sides(1 - i_dim_bias);
				if(not box.is_wall(pos_i)) {
					int i_dim_hit = 1 - i_dim_bias;
					i_dims_hit = Range(i_dim_hit, i_dim_hit);
				} else {
					int i_dim_nonbias = 1 - i_dim_bias;
					pos_i = test_i;
					pos_i(i_dim_nonbias) += sides(1 - i_dim_nonbias);
					if (not box.is_wall(pos_i)) {
						int i_dim_hit = 1 - i_dim_nonbias;
						i_dims_hit = Range(i_dim_hit, i_dim_hit);
					} else {
//						i_dims_hit = Range::all();
						i_dims_hit = Range(0, 1);
					}
				}
			} else {
				cout << "Error: Update rs went wrong (invalid adjacentness: " << adjacentness << " )." << endl;
			}
			test_r(i_dims_hit) = test_cell_r(i_dims_hit) + sides(i_dims_hit) *
				((box.dx_get() / 2.0) + box.cell_buffer_get());
			wall_specular(i_arrow, i_dims_hit);
		}
		m_rs(i_arrow, Range::all()) = test_r;
	}
}

void Arrows::vs_update(void) {
	tumble();
}

void Arrows::rates_update(const Box& box) {
	(this->*m_prates_update)(box);
}

void Arrows::wall_handle(int i_arrow, Range i_dims_hit) {
	(this->*m_pwall_handle)(i_arrow, i_dims_hit);
}

const Array<float, DIM>& Arrows::rs_get(void) const {
	return m_rs;
}

const Array<float, DIM>& Arrows::vs_get(void) const {
	return m_vs;
}

int Arrows::num_arrows_get(void) const {
	return m_num_arrows;
}

void Arrows::rs_initialise(const Box& box) {
	m_rs.resize(m_num_arrows, DIM);
	float L_half = box.L_get() / 2.0;
	for(int i_arrow = 0; i_arrow < m_num_arrows; i_arrow++) {
		for(int i_dim = 0; i_dim < DIM; i_dim++)
			m_rs(i_arrow, i_dim) = random_float(-L_half, +L_half);
		while(box.is_wall(m_rs(i_arrow, Range::all())))
			for(int i_dim = 0; i_dim < DIM; i_dim++)
				m_rs(i_arrow, i_dim) = random_float(-L_half, +L_half);
	}
}

void Arrows::vs_initialise() {
	m_vs.resize(m_num_arrows, DIM);
	Array<float, 1> v_p(DIM);
	v_p(0) = m_v;
	for(int i_arrow = 0; i_arrow < m_num_arrows; i_arrow++) {
		v_p(1) = random_float(-M_PI, +M_PI);
		m_vs(i_arrow, Range::all()) = polar_to_cart(v_p);
	}
}

void Arrows::rates_initialise(char rate_alg, float sense_mem, float t_mem) {
	m_rates.resize(m_num_arrows);

	if (rate_alg == 'm') {
		float A = 0.5;
		float N = pow((0.8125 * pow(A, 2) - 0.75 * A + 0.5), -0.5);
		int i_t__max = static_cast<int>(t_mem / DELTA_t);
		Array<float, 1> g(i_t__max);
		g = m_rate_base * i1 * DELTA_t;
		m_K.resize(g.shape());
		m_K = N * sense_mem * m_rate_base * exp(-g) * (1 - A * (g + pow2(g) / 2.0));
		m_attract_mem.resize(m_num_arrows, i_t__max);
	}
}

void Arrows::tumble(void) {
	Array<float, 1> v_p(DIM), v_c(DIM);
	for(int i_arrow = 0; i_arrow < m_num_arrows; i_arrow++) {
		float dice_roll = random_float(0.0, 1.0);
		if(dice_roll < (m_rates(i_arrow) * DELTA_t)) {
			v_c = m_vs(i_arrow, Range::all());
			v_p = cart_to_polar(v_c);
			v_p(1) = random_float(-M_PI, +M_PI);
			m_vs(i_arrow, Range::all()) = polar_to_cart(v_p);
		}
	}
}

float Arrows::grad_find(const Box& box, Array<float, DIM>& field, int i_arrow) {
	TinyVector<int, DIM> arrow_i = box.r_to_i(m_rs(i_arrow, Range::all()));

	Array<float, 1> v_abs(abs(m_vs(i_arrow, Range::all())));
	int i_dim_bias = argmax(v_abs);
	int bias_sign = static_cast<int>(copysign(1.0, m_vs(i_arrow, i_dim_bias)));

	TinyVector<int, DIM> pos_i(arrow_i);
	pos_i(i_dim_bias) = arrow_i(i_dim_bias) + bias_sign;
	float grad = field(pos_i(0), pos_i(1));
	pos_i(i_dim_bias) = arrow_i(i_dim_bias) - bias_sign;
	grad -= field(pos_i(0), pos_i(1));
	return grad / box.dx_get();
}

float Arrows::integral_find(int i_arrow) {
	return sum(m_attract_mem(i_arrow, Range::all()) * m_K * DELTA_t);
}

void Arrows::rates_update_const(const Box& box) {
		m_rates = m_rate_base;
}

void Arrows::rates_update_grad(const Box& box) {
	Array<float, DIM> attract = box.attract_get();
	for(int i_arrow = 0; i_arrow < m_num_arrows; i_arrow++) {
		m_rates(i_arrow) =
				m_rate_base * (1.0 -
				m_sense_grad * grad_find(box, attract, i_arrow));
	}
}

void Arrows::rates_update_mem(const Box& box) {
	TinyVector<int, DIM> arrow_i;
	const Array<float, DIM>& attract = box.attract_get();
	m_attract_mem(Range::all(), Range(1, toEnd)) =
			m_attract_mem(Range::all(), Range(fromStart, m_attract_mem.extent(secondDim) - 2));
	for(int i_arrow = 0; i_arrow < m_num_arrows; i_arrow++) {
		arrow_i = box.r_to_i(m_rs(i_arrow, Range::all()));
		m_attract_mem(i_arrow, 0) = attract(arrow_i);
		m_rates(i_arrow) = m_rate_base * (1.0 - integral_find(i_arrow));
	}
}

void Arrows::wall_specular(int i_arrow, Range i_dims_hit) {
	m_vs(i_arrow, i_dims_hit) *= -1.0;
}

void Arrows::wall_bounceback(int i_arrow, Range i_dims_hit) {
	m_vs(i_arrow, Range::all()) *= -1.0;
}

void Arrows::wall_aligning(int i_arrow, Range i_dims_hit) {
	Array<float, 1> direction(DIM), v(DIM);
	direction(i_dims_hit) = 0.0;
	float d_mag = vector_mag(direction);
	if(d_mag < ZERO_THRESH) {
		int sign = copysign(1.0, random_float(-1.0, +1.0));
		direction = m_vs(i_arrow, Range::all()).copy();
		for (int i_dim = 0; i_dim < DIM; i_dim++)
			m_vs(i_arrow, i_dim) = sign * direction(1 - i_dim);
	} else {
		v = m_vs(i_arrow, Range::all());
		float v_mag = vector_mag(v);
		m_vs(i_arrow, Range::all()) = direction * (v_mag / d_mag);
	}
}

void Arrows::wall_stalling(int i_arrow, Range i_dims_hit) {
	m_vs(i_arrow, i_dims_hit) = 0.0;
}
