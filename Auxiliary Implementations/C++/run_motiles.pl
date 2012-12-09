#!/usr/local/bin/perl -w
open (GP, "|/usr/bin/gnuplot -persist") or die "no gnuplot";
use FileHandle;
GP->autoflush(1);
for ($fnum = 1; $fnum < 101; $fnum++) {	
	$fname=sprintf("macey_vectors_%05d.dat", $fnum);
	print GP "set term x11;\
plot '/home/ejm/workspace/macey/dat/macey_lattice.dat' matrix with image,'/home/ejm/workspace/macey/dat/$fname' using 1:2:3:4 with vectors\n";
	select(undef, undef, undef, 0.1);
}
close GP
