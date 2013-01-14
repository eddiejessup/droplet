import numpy as np
import utils
import fields
import walls as walls_module
import walled_fields
import motiles

class System(object):
    def __init__(self, args):
        self.init_general(args['general'])
        self.init_fields(args['fields'])
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
        self.i = 0.0

    def init_fields(self, args):
        if args['obstructions_alg'] == 'none':
            self.o = walls_module.Walls(self, args['dx'])
        elif args['obstructions_alg'] == 'traps':
            trap_args = args['obstructions']['traps']
            self.o = walls_module.Traps(self, args['dx'], trap_args['n'], trap_args['thickness'], trap_args['width'], trap_args['slit_width'])
        elif args['obstructions_alg'] == 'maze':
            maze_args = args['obstructions']['maze']
            self.o = walls_module.Maze(self, args['dx'], maze_args['thickness'], maze_args['seed'])
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
        self.i += 1

    def get_A(self):
        return self.L ** self.dim

    def get_dstd(self, dx):
        density = self.m.get_density_field(dx)
        valids = np.asarray(np.logical_not(self.o.to_field(dx), dtype=np.bool))
        return np.std(density[valids])