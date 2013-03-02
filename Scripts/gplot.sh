#! /bin/bash

echo "plot '$1' u 1:2 w lp" | gnuplot -p
