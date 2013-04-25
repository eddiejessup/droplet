import matplotlib as mpl
import matplotlib.mlab as mlb
import matplotlib.pyplot as pp

mpl.rc('font', family='serif', serif='STIXGeneral')

fnames=[8,11,16,23,32,45,64]

for f in fnames:
	d = mlb.csv2rec('%s/d.csv' % f, delimiter=' ')
	pp.errorbar(d['vf'], d['acc'], yerr=d['acc_err'], label='R=%s' % f)

pp.xscale('log')
pp.legend()
pp.xlabel(r'$\mathrm{V_p} / \mathrm{V_d}$', fontsize=20)
pp.ylabel(r'$(r-r_0) / \mathrm{R}$', fontsize=20)
pp.show()
