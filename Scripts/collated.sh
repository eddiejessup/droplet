# /bin/bash

drophist.py "ajdslkfghfghfdkgjf" -t
for d in "$@"; do
	drophist.py -pt $d/* > $d/d.csv
	drophist.py $d/*
done
