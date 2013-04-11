#! /bin/bash

for d in $@; do
	for d2 in $d/*; do
		mv $d2/b.png "$d2.png"
		rmdir $d2
	done
done
