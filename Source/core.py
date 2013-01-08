import os
import shutil
import sys
import yaml
import numpy as np
import matplotlib.pyplot as pp
from mpl_toolkits.mplot3d import Axes3D
import utils
import fields
import walls as walls_module
import walled_fields
import motiles

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

class System(object):
    def __init__(self, params_fname):
        args = yaml.safe_load(open(params_fname, 'r'))
        self.init_general(args['general'])
        self.init_field(args['field'])
        self.init_motiles(args['motiles'])

    def init_general(self, args):
        self.seed = args['seed']
        self.dt = args['dt']
        self.dim = args['dimension']
        self.L = args['L']

        if self.L <= 0.0:
            raise Exception('Require system size > 0')
        if self.dt <= 0.0:
            raise Exception('Require time-step > 0')
        if self.dim < 0:
            raise Exception('Require dimension >= 0')

        np.random.seed(self.seed)
        self.L_half = self.L / 2.0
        self.t = 0.0
        self.i_t = 0.0

    def init_field(self, args):
        if args['obstructions_alg'] == 'none':
            self.o = walls_module.Walls(self, args['dx'])
        elif args['obstructions_alg'] == 'traps':
            trap_args = args['obstructions']['traps']
            self.o = walls_module.Traps(self, args['dx'], trap_args['n'], trap_args['thickness'], trap_args['width'], trap_args['slit_width'])
        elif args['obstructions_alg'] == 'maze':
            maze_args = args['obstructions']['maze']
            self.o = walls_module.Maze(self, args['dx'], maze_args['width'], maze_args['seed'])
        elif args['obstructions_alg'] == 'parametric':
            para_args = args['obstructions']['parametric']
            self.o = walls_module.Parametric(self, para_args['n'], para_args['stickiness'], para_args['radius_min'], para_args['radius_max'])
        else:
            raise Exception('Invalid obstruction algorithm')

        if args['f_pde_flag']:
            f_pde_args = args['f_pde']
            self.f = walled_fields.Food(self, args['dx'], self.o, f_pde_args['D'], f_pde_args['sink_rate'], a_0=args['f_0'])
        else:
            self.f = walled_fields.Scalar(self, args['dx'], self.o, a_0=args['f_0'])

        if args['c_pde_flag']:
            c_pde_args = args['c_pde']
            self.c = walled_fields.Secretion(self, args['dx'], self.o, c_pde_args['D'], c_pde_args['sink_rate'], c_pde_args['source_rate'], a_0=0.0)
        else:
            self.c = walled_fields.Scalar(self, args['dx'], self.o, a_0=0.0)

    def init_motiles(self, args):
        if self.dim == 1:
            num_motiles = int(self.o.get_A_free() * args['density_1d'])
        elif self.dim == 2:
            num_motiles = int(self.o.get_A_free() * args['density_2d'])
        elif self.dim == 3:
            num_motiles = int(self.o.get_A_free() * args['density_3d'])
        tumble_args = args['tumble'] if args['tumble_flag'] else None
        force_args = args['force'] if args['force_flag'] else None
        rot_diff_args = args['rot_diff'] if args['rot_diff_flag'] else None
        vicsek_args = args['vicsek'] if args['vicsek_flag'] else None

        self.m = motiles.Motiles(self, num_motiles, args['v_0'], self.o, args['tumble_flag'], tumble_args,
            args['force_flag'], force_args, args['rot_diff_flag'], rot_diff_args, args['vicsek_flag'], vicsek_args)

    def iterate(self):
        self.m.iterate(self.c, self.o)
        density = self.m.get_density_field(self.f.dx)
        self.f.iterate(density)
        self.c.iterate(density, self.f)
        self.t += self.dt
        self.i_t += 1

    def output_persistent(self, dirname, prefix=''):
        file = open('%s/%sparams.dat' % (dirname, prefix), 'w')
        file.write('seed,%i\n' % self.seed)
        file.write('dt,%f\n' % self.dt)
        file.write('dim,%i\n' % self.dim)
        file.write('L,%f\n' % self.L)
        file.close()
        self.o.output_persistent(dirname, prefix=prefix+'o_')
        self.f.output_persistent(dirname, prefix=prefix+'f_')
        self.c.output_persistent(dirname, prefix=prefix+'c_')
        self.m.output_persistent(dirname, prefix=prefix+'m_')

#        self.fig = pp.figure()
#        self.lims = [-self.L_half, self.L_half]
#        if self.dim == 3:
#            self.ax = self.fig.add_subplot(111, projection='3d')
#            self.parts_plot = self.ax.scatter([], [], [])
#            self.ax.set_zlim(self.lims)
#            self.ax.set_zticks([])
#            self.ax.set_xlim(self.lims)
#            self.ax.set_ylim(self.lims)
#            self.ax.set_xticks([])
#            self.ax.set_yticks([])
#            self.ax.set_aspect('equal')

    def output(self, dirname, prefix=''):
        file = open('%s/%sstate.dat' % (dirname, prefix), 'w')
        file.write('t,%f\n' % self.t)
        file.write('i_t,%i\n' % self.i_t)
        file.close()
        self.m.output(dirname, prefix=prefix+'m_')
        self.f.output(dirname, prefix=prefix+'f_')
        self.c.output(dirname, prefix=prefix+'c_')
#        if self.dim == 2:
#            pp.scatter(self.m.r[:, 0], self.m.r[:, 1], s=2)
##            pp.imshow(np.ma.array(self.c.a.T, mask=self.c.of.T), extent=2*[-self.L_half, self.L_half], origin='lower')
##            pp.colorbar()
#            pp.xlim(self.lims)
#            pp.ylim(self.lims)
#            pp.xticks([])
#            pp.yticks([])
#            pp.savefig('%s/%s.png' % (dirname, self.i_t))
#            pp.clf()
#        elif self.dim == 3:
#            self.parts_plot._offsets3d = (self.m.r[..., 0], self.m.r[..., 1], self.m.r[..., 2])
#            pp.savefig('%s/%s.png' % (dirname, self.i_t))
#        print(self.t, self.m.N)

    def get_A(self):
        return self.L ** self.dim

def main():
    print('Initialising...')

    # Get parameters
    params_fname = sys.argv[1]
    args = yaml.safe_load(open(params_fname, 'r'))['system']

    # Initialise environment
    system = System(params_fname)

    # Make output directory if it isn't there already
    if args['output_flag']:
        utils.make_dirs_safe(args['output']['path'])
        shutil.copy(params_fname, args['output']['path'])
        utils.make_dirs_safe('%s/Persistent' % args['output']['path'])
        system.output_persistent('%s/Persistent/' % args['output']['path'])

    print('Initialisation done! Starting...')
    while system.t < args['t_max']:
        system.iterate()
        if args['output_flag'] and not system.i_t % args['output']['every']:
            state_dirname = '%s/%f' % (args['output']['path'], system.t)
            utils.make_dirs_safe(state_dirname)
            system.output(state_dirname)

    print('Finished!')

if __name__ == '__main__':
    main()
#    import cProfile; cProfile.run('main()', sort='cum')
