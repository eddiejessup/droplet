/*
 * Box.cpp
 *
 *  Created on: 2 Dec 2011
 *      Author: ejm
 */

#include "Box.hpp"

Box::Box(float L, int lattice_res, float cell_buffer,
				 float D_attract, float attract_rate, float breakdown_rate,
				 float food_0,
				 bool food_pde_flag, float D_food, float metabolism_rate,
				 int density_range) {
	m_L = L;
	m_cell_buffer = cell_buffer;
	m_attract_rate = attract_rate;
	m_breakdown_rate = breakdown_rate;
	m_food_pde_flag = food_pde_flag;
	m_metabolism_rate = metabolism_rate;
	m_density_range = density_range;

	walls_initialise(lattice_res);
	walls_find();

	m_M = m_walls.extent(firstDim);
	m_dx = L / static_cast<float>(m_M);

	m_density.resize(m_walls.shape());
	attract_initialise(D_attract);
	food_initialise(food_0, D_food);
}

bool Box::is_wall(const TinyVector<int, DIM>& i) const {
	return m_walls(i);
}

bool Box::is_wall(const TinyVector<float, DIM>& r) const {
	return m_walls(r_to_i(r));
}

bool Box::is_wall(const Array<float, 1>& r) const {
	return m_walls(r_to_i(r));
}

TinyVector<int, DIM> Box::r_to_i(const TinyVector<float, DIM>& r) const {
	return ((r + m_L / 2.0) / m_dx);
}

TinyVector<int, DIM> Box::r_to_i(const Array<float, 1>& r) const {
	TinyVector<int, DIM> i;
	for (int i_dim = 0; i_dim < DIM; i_dim++)
		i(i_dim) = ((r(i_dim) + m_L / 2.0) / m_dx);
	return i;
}

Array<float, 1> Box::i_to_r(const TinyVector<int, DIM>& i) const {
	Array<float, 1> r(DIM);
	for (int i_dim = 0; i_dim < DIM; i_dim++)
		r(i_dim) = -(m_L / 2.0) + (i(i_dim) + 0.5) * m_dx;
	return r;
}

void Box::fields_update(const Array<float, DIM>& arrow_rs) {
	density_update(arrow_rs);
	attract_update();
	if(m_food_pde_flag)
		food_update();
}

const Array<bool, DIM>& Box::walls_get(void) const {
	return m_walls;
}

const Array<float, DIM>& Box::attract_get(void) const {
	return m_attract;
}

float Box::dx_get(void) const {
	return m_dx;
}

float Box::L_get(void) const {
	return m_L;
}

float Box::cell_buffer_get(void) const {
	return m_cell_buffer;
}

// Private

void Box::walls_initialise(int wall_res) {
	wall_res = 2 * (wall_res / 2) + 1;
	m_walls.resize(wall_res, wall_res);
	m_walls(Range::all(), 0) = true;
	m_walls(Range::all(), wall_res - 1) = true;
	m_walls(wall_res - 1, Range::all()) = true;
	m_walls(0, Range::all()) = true;
}

void Box::walls_find(void) {
    int i_1_8 = m_walls.extent(firstDim) / 8;
    int i_3_8 = 3 * i_1_8;
    int i_4_8 = 4 * i_1_8;
    int i_5_8 = 5 * i_1_8;
    m_walls(Range(i_3_8, i_5_8), i_3_8) = true;
    m_walls(Range(i_3_8, i_5_8), i_5_8) = true;
    m_walls(i_3_8, Range(i_3_8, i_5_8)) = true;
    m_walls(i_5_8, Range(i_3_8, i_5_8)) = true;

    m_walls(i_4_8, i_5_8) = false;
}

void Box::attract_initialise(float D) {
	m_attract.resize(m_walls.shape());
	m_attract_coeff_const = D * DELTA_t / pow(m_dx, 2);
}

void Box::food_initialise(float food_0, float D) {
	m_food.resize(m_walls.shape());
	m_food = 1 - m_walls;
	if (m_food_pde_flag)
		m_food_coeff_const = D * DELTA_t / pow(m_dx, 2);
}

void Box::density_update(const Array<float, DIM>& arrow_rs) {
	m_density = 0.0;
	for (int i_arrow = 0; i_arrow < arrow_rs.extent(secondDim); i_arrow++) {
		TinyVector<int, DIM> i;
		i = r_to_i(arrow_rs(i_arrow, Range::all()));
		for (int i_off_x = -m_density_range; i_off_x <= +m_density_range; i_off_x++) {
			int i_x = i(0) + i_off_x;
			if ((i_x >= 0) and (i_x < m_M))
				for (int i_off_y = -m_density_range; i_off_y <= +m_density_range; i_off_y++) {
					int i_y = i(1) + i_off_y;
					if ((i_y >= 0) and (i_y < m_M) and (not m_walls(i_x, i_y)))
							m_density(i_x, i_y)++;
				}
		}
	}
}

void Box::food_update(void) {
	if(m_food_pde_flag) {
		m_food -= DELTA_t * m_metabolism_rate * m_density;
		lattice_diffuse(m_food, m_food_coeff_const);
		for(int i_x = 0; i_x < m_M; i_x++)
			for(int i_y = 0; i_y < m_M; i_y++)
				if(m_food(i_x, i_y) < 0.0)
					m_food(i_x, i_y) = 0.0;
	}
}

void Box::attract_update(void) {
	lattice_diffuse(m_attract, m_attract_coeff_const);
	m_attract += DELTA_t * (
			m_attract_rate * m_density * m_food -
			m_breakdown_rate * m_attract);
	for(int i_x = 0; i_x < m_M; i_x++)
		for(int i_y = 0; i_y < m_M; i_y++)
			if (m_attract(i_x, i_y) < 0.0) m_attract(i_x, i_y) = 0.0;
}

void Box::lattice_diffuse(Array<float, DIM>& field, float coeff) {
	Array<float, DIM> field_coeff_arr(field.shape());
	field_coeff_arr = 0.0;
	for (int i_x = 1; i_x < m_M - 1; i_x++)
		for (int i_y = 1; i_y < m_M - 1; i_y++) {
			float field_cur = field(i_x, i_y);
			field_coeff_arr(i_x, i_y) = (not m_walls(i_x, i_y)) *
				((not m_walls(i_x + 1, i_y)) * (field(i_x + 1, i_y) - field_cur) +
				 (not m_walls(i_x - 1, i_y)) * (field(i_x - 1, i_y) - field_cur) +
				 (not m_walls(i_x, i_y + 1)) * (field(i_x, i_y + 1) - field_cur) +
				 (not m_walls(i_x, i_y - 1)) * (field(i_x + 1, i_y) - field_cur));
		}
	field += coeff * field_coeff_arr;
}
