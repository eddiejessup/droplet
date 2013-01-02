import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as pp
import fields
import walls as walls_module
import walled_fields
import motiles

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

class Box(object):
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

        np.random.seed(self.seed)
        self.L_half = self.L / 2.0
        self.t = 0.0
        self.i_t = 0.0

    def init_field(self, args):
        if args['obstructions_alg'] == 'none':
            para_args = args['obstructions']['parametric']
            self.o = walls_module.Walls(self, args['dx'])
        elif args['obstructions_alg'] == 'traps':
            trap_args = args['obstructions']['traps']
            self.o = walls_module.Traps(self, args['dx'], trap_args['n'], trap_args['thickness'], trap_args['width'], trap_args['slit_width'])
        elif args['obstruction_alg'] == 'maze':
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
        num_motiles = int(self.o.get_A_free() * args['density'])
        tumble_args = args['tumble'] if args['tumble_flag'] else None
        force_args = args['force'] if args['force_flag'] else None
        noise_args = args['noise'] if args['noise_flag'] else None
        vicsek_args = args['vicsek'] if args['vicsek_flag'] else None

        self.motiles = motiles.Motiles(self, num_motiles, args['v_0'], args['tumble_flag'], tumble_args, 
            args['force_flag'], force_args, args['noise_flag'], noise_args, args['vicsek_flag'], vicsek_args)
        self.o.init_r(self.motiles)

    def iterate(self):
        self.motiles.iterate(self.c)
        self.o.iterate_r(self.motiles)
        density = self.motiles.get_density_field(self.f.dx)
        self.f.iterate(density)
        self.c.iterate(density, self.f)
        self.t += self.dt
        self.i_t += 1

    def output(self, dirname):
        pp.scatter(self.motiles.r[:, 0], self.motiles.r[:, 1], s=20)
        pp.imshow(np.ma.array(self.c.a.T, mask=self.c.of.T), extent=2*[-self.L_half, self.L_half], origin='lower')
        pp.colorbar()
        pp.savefig('%s/%s.png' % (dirname, self.i_t))
        pp.clf()
        print(self.t, self.f.a.mean())

    def get_A(self):
        return self.L ** self.dim

def main():
    print('Starting!')
    params_fname = sys.argv[1]
    args = yaml.safe_load(open(params_fname, 'r'))['system']
    box = Box(params_fname)
    while box.t < args['t_max']:
        box.iterate()
        if args['output_flag'] and not box.i_t % args['output']['every']:
            box.output(args['output']['path'])
    print('Done!')

if __name__ == '__main__': 
#    main()
    import cProfile; cProfile.run('main()', sort='cum')
