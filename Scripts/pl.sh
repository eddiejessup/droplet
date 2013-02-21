#! /bin/bash

c="plot "
for f in $1*/log.csv; do
    c+="'$f' u 't':'$2' w lp ti '${f%/log.csv}', "
done
echo ${c%, } | gnuplot -p