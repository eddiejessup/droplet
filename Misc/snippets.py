drophist.py -h `find -type d | grep "/r" | sed s_/r__`

plot for [f in "1 2 5 10 20"] f."/log.csv" u 1:"v_drift":"v_drift_err" w yerrorbars

mencoder mf://@fnames.txt -mf w=800:h=600:fps=50:type=png -ovc x264 -x264encopts crf=25 -oac copy -o output.avi

# Sort a directory numerically, number starts at 5th character
ls dyn/*.npz | sort -n -k1.5n

# If there are arguments, use those, otherwise expect to be fed data
f = sys.argv[2:] if sys.argv[2:] else sys.stdin