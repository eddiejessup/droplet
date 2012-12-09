/*
 * main.cpp
 *
 *  Created on: 3 Dec 2011
 *      Author: ejm
 */

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "blitz/array.h"

#include "utils.hpp"
#include "Box.hpp"
#include "Arrows.hpp"

//#define BZ_DEBUG

using namespace std;
using namespace blitz;

int main() {
	const float		t_MAX = 200 * DELTA_t;

	const float 	L = 100.0;
	const int 		LATTICE_RES = 50;
	const float 	CELL_BUFFER = 1e-8;

	const float		D_ATTRACT = 0.1,
								ATTRACT_RATE = 10.0,
								BREAKDOWN_RATE = 0.01;

	const float 	FOOD_0 = 1.0;

	const bool		FOOD_PDE_FLAG = false;
	const float		D_FOOD = 0.05,
								METABOLISM_RATE = 1.0;

	const int 		DENSITY_RANGE = 1;

	const int			NUM_ARROWS = 1000;

	const float		SENSE_GRAD = 0.01;

	const float		t_MEM = 5.0,
								SENSE_MEM = 0.05;

	const bool 		ONESIDED_FLAG = true;
	const char		RATE_ALG = 'm',
								BC_ALG = 'a';

	Box box(L, LATTICE_RES, CELL_BUFFER,
					D_ATTRACT, ATTRACT_RATE, BREAKDOWN_RATE,
					FOOD_0,
					FOOD_PDE_FLAG, D_FOOD, METABOLISM_RATE,
					DENSITY_RANGE);

	Arrows arrows(box, NUM_ARROWS,
								SENSE_GRAD,
								t_MEM, SENSE_MEM,
								ONESIDED_FLAG, RATE_ALG, BC_ALG);

	ofstream file;

	file.open("./dat/macey_lattice.dat");
	file << "# Macey lattice" << endl;
	file << "# v i_y v   > i_x >" << endl;
	const Array<bool, DIM>& ws = box.walls_get();
	for (int i_y = 0; i_y < ws.extent(firstDim); i_y++) {
		file << ws(0, 0);
		for (int i_x = 1; i_x < ws.extent(secondDim); i_x++) {
			file << " " << ws(i_x, i_y);
		}
		file << endl;
	}
	file.close();

	int iter_count = 1;
	std::stringstream ss;

	const float ARROW_SIZE = 0.02;
	const int EVERY = 5;

	for(float t = 0.0; t < t_MAX; t += DELTA_t) {
//		if(iter_count % EVERY == 0) {
//			ss.str(string());
//			ss << "./dat/macey_vectors_" << setw(5) << std::setfill('0') << iter_count / EVERY << ".dat";
//			cout << (ss.str()).c_str() << endl;
//			file.open((ss.str()).c_str());
//
//			Array<float, DIM> rs_pl = (arrows.rs_get()).copy();
//			Array<float, DIM> vs_pl = (arrows.vs_get()).copy();
//			rs_pl = (rs_pl + box.L_get() / 2.0) * ws.extent(firstDim);
//			vs_pl *= ARROW_SIZE;
//
//			file << "# Macey vectors" << endl;
//			file << "# r_x r_y v_x v_y" << endl;

//			for(int i_arrow = 0; i_arrow < NUM_ARROWS; i_arrow++) {
//				float n = vector_mag(vs_pl(i_arrow, Range::all()));
//				file << rs_pl(i_arrow, 0) << " " << rs_pl(i_arrow, 1) << " " <<
//						vs_pl(i_arrow, 0) / n << " " << vs_pl(i_arrow, 1) / n << endl;
//			}
//			file.close();
//		}
		iter_count++;
		cout << iter_count << endl;

		arrows.rs_update(box);
		box.fields_update(arrows.rs_get());
		arrows.vs_update();
		arrows.rates_update(box);
	}
	cout << "Done." << std::endl;
	return 1;
}
