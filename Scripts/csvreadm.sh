#! /bin/bash

echo "chi $2 $2_err"
for f in $1*/log.csv; do
    echo -ne "${f%/log.csv} "
    csvread.py $f $2 $3 $4
done