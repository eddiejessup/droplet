import model

# n_10_v_13.5_l_1.23_R_0.36_D_0.25_Dr_0.05_Rd_16.0_Drc_10
v = 13.5
l = 1.23
R = 0.36
D = 0.25
Dr = 0.05
R_d = 16.0
dim = 3
t_max = 10000.0
dt = 0.01
every = 100
Dr_c = 10.0
align = True

n = 10
out = 'out'
model.dropsim(n, v, l, R, D, Dr, R_d, dim, t_max, dt, out, every, Dr_c,
              align=align, tracking=True)
