#! /bin/bash

echo chi v_drift v_drift_err
for f in $1*/log.csv; do
    echo -ne "${f%/log.csv} "
    csvread.py $f $2 $3
done