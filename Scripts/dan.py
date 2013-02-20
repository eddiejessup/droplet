import sys
import csv
import numpy as np

pfs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45]
assert len(pfs) == len(sys.argv[1:])
results = csv.writer(open('res.csv', 'w'), delimiter=' ')
results.writerow(['pf', 'D_mean', 'D_var'])
for fname, pf in zip(sys.argv[1:], pfs):
    reader = csv.reader(open(fname, 'r'), delimiter=' ')
    Ds = []
    header = True
    for row in reader:
        if not header:
            Ds.append(float(row[2]))
        header = False
    Ds = Ds[int(round(0.25*len(Ds))):]
    D_mean = np.mean(Ds)
    D_var = np.var(Ds)
    print 'fname %s, D: %f plusminus %f' % (fname, D_mean, D_var)
    results.writerow([pf, D_mean, D_var])
