filler = np.frompyfunc(lambda x: list(), 1, 1)
cl = np.empty([M, M], dtype=np.object)
filler(cl, cl)
inds = utils.r_to_i(r, dx)
for i in range(len(inds)):
    cl[inds[i, 0], inds[i, 1]].append(i)
