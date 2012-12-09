/*
 * Arrows.hpp
 *
 *  Created on: 3 Dec 2011
 *      Author: ejm
 */

#ifndef BOX_HPP
#define BOX_HPP

#include <math.h>

#include "blitz/array.h"

#include "utils.hpp"

using blitz::Array;

class Box {
public:
	Box(float L, int lattice_res, float cell_buffer,
			float D_attract, float attract_rate, float breakdown_rate,
			float food_0,
			bool food_pde_flag, float D_food, float metabolism_rate,
			int density_range);

	bool is_wall(const TinyVector<int, DIM>& i) const;
	bool is_wall(const TinyVector<float, DIM>& r) const;
	bool is_wall(const Array<float, 1>& r) const;
	TinyVector<int, DIM> r_to_i(const TinyVector<float, DIM>& r) const;
	TinyVector<int, DIM> r_to_i(const Array<float, 1>& r) const;
	Array<float, 1> i_to_r(const TinyVector<int, DIM>& i) const;
	void fields_update(const Array<float, 2>& rs);
	const Array<bool, DIM>& lattice_get(void) const;
	const Array<float, DIM>& attract_get(void) const;
	float dx_get(void) const;
	float L_get(void) const;
	float cell_buffer_get(void) const;
	const Array<bool, DIM>& walls_get(void) const;

private:
	void walls_initialise(int wall_res);
	void walls_find(void);
	void attract_initialise(float D);
	void food_initialise(float food_0, float D);
	void density_update(const Array<float, DIM>& rs);
	void food_update(void);
	void attract_update(void);
	void lattice_diffuse(Array<float, DIM>& field, float coeff);

	int m_M;
	float m_L, m_cell_buffer, m_dx;
	Array<bool, DIM> m_walls;

	int m_density_range;
	Array<int, DIM> m_density;

	float m_attract_rate, m_breakdown_rate, m_attract_coeff_const;
	Array<float, DIM> m_attract;

	bool m_food_pde_flag;
	float m_food_coeff_const, m_metabolism_rate;
	Array<float, DIM> m_food;
};

#endif // BOX_HPP
