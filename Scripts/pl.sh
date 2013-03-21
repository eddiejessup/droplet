#! /bin/bash

c="plot "
for f in $2*/log.csv; do
    c+="'$f' u 't':'$1':'$1_err'"
    if [ $# -gt 2 ]; then
        c+=" w $3 "
    else
        c+="w yerrorbars "
    fi
    c+="ti '${f%/log.csv}'"
    c+=", "
done
echo ${c%, } | gnuplot -p