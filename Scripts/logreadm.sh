#! /bin/bash

echo "x $2 $2_err"
for f in $1*/log.csv; do
    echo -ne "${f%/log.csv} "
    logread.py $f $2
done