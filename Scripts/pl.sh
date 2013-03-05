#! /bin/bash

c="plot "
for f in $1*/log.csv; do
    c+="'$f' u 't':'$2':'$2_err' w yerrorbars ti '${f%/log.csv}'"
    if [ $# -gt 2 ]; then
        c+=" w $3"
    fi
    c+=", "
done
echo ${c%, } | gnuplot -p