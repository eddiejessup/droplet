#!/usr/bin/env python

import os
import time
import argparse
import time
import numpy as np
import vtk
from vtk.util import numpy_support
import utils
import butils

def pad_to_3d(a):
    a_pad = np.zeros([len(a), 3], dtype=a.dtype)
    a_pad[:, :a.shape[-1]] = a
    return a_pad

parser = argparse.ArgumentParser(description='Visualise system states using VTK')
parser.add_argument('dyns', nargs='*',
    help='npz files containing dynamic states')
parser.add_argument('-s', '--save', default=False, action='store_true',
    help='Save plot')
parser.add_argument('-c', '--cross', type=float, default=1.0,
    help='Cross section fraction')
args = parser.parse_args()

multis = len(args.dyns) > 1
if multis:
    dt = butils.t(args.dyns[1]) - butils.t(args.dyns[0])

datdir = os.path.abspath(os.path.join(args.dyns[0], '../..'))

# create a rendering window and renderer
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.SetSize(600, 600)
renWin.AddRenderer(ren)
if args.save:
    # Hide onscreen render window
    renWin.OffScreenRenderingOn()

    # create a window to image filter attached to render window
    winImFilt = vtk.vtkWindowToImageFilter()
    winImFilt.SetInput(renWin)

    writer = vtk.vtkPNGWriter()
    writer.SetInputConnection(winImFilt.GetOutputPort())
else:
    # create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.Initialize()

stat = butils.get_stat(datdir)
sysenv = butils.get_env(datdir)
lu = sysenv.p.lu
ld = sysenv.p.ld
R = sysenv.p.R

# System bounds
L = stat['L']
# Poly
sys = vtk.vtkCubeSource()
sys.SetXLength(L)
sys.SetYLength(L)
sys.SetZLength(L)
# Mapper    
sysMapper = vtk.vtkPolyDataMapper()
sysMapper.SetInputConnection(sys.GetOutputPort())
# Actor
sysActor = vtk.vtkActor()
sysActor.GetProperty().SetOpacity(0.2)
sysActor.SetMapper(sysMapper)
# ren.AddActor(sysActor)

timeActor = vtk.vtkTextActor()
timeActor.SetInput('init')
ren.AddActor(timeActor)

# Obstacles
# Mapper
envMapper = vtk.vtkPolyDataMapper()
envActor = vtk.vtkActor()

if 'o' in stat:
    o = stat['o']

    dx = L / o.shape[0]
    inds = np.indices(o.shape).reshape((o.ndim, -1)).T
    ors = []
    for ind in inds:
        if o[tuple(ind)]:
            ors.append(ind)
    ors = np.array(ors) * dx - L / 2.0
    ors = pad_to_3d(ors)

    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(ors))
    polypoints = vtk.vtkPolyData()
    polypoints.SetPoints(points)

    cubeSource = vtk.vtkCubeSource()
    cubeSource.SetXLength(dx)
    cubeSource.SetYLength(dx)
    cubeSource.SetZLength(dx)

    env = vtk.vtkGlyph3D()
    env.SetSourceConnection(cubeSource.GetOutputPort())
    env.SetInputData(polypoints)

elif 'r' in stat:
    r_pack = stat['r']
    R_pack = stat['R']

    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(r_pack))
    polypoints = vtk.vtkPolyData()
    polypoints.SetPoints(points)

    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetThetaResolution(30)
    sphereSource.SetPhiResolution(30)
    sphereSource.SetRadius(R_pack)

    env = vtk.vtkGlyph3D()
    env.SetSourceConnection(sphereSource.GetOutputPort())
    env.SetInputData(polypoints)

elif 'R' in stat:
    R_drop = stat['R']

    env = vtk.vtkSphereSource()
    env.SetThetaResolution(30)
    env.SetPhiResolution(30)
    env.SetRadius(R_drop)

try:
    envMapper.SetInputConnection(env.GetOutputPort())
except NameError:
    pass
else:
    envActor.SetMapper(envMapper)
    envActor.GetProperty().SetColor(0, 1, 0)
    envActor.GetProperty().SetRepresentationToWireframe()
    # envActor.GetProperty().SetOpacity(0.5)
    ren.AddActor(envActor)

particleCPoints = vtk.vtkPoints()
particleCPolys = vtk.vtkPolyData()
particleCPolys.SetPoints(particleCPoints)
particlesC = vtk.vtkGlyph3D()

lineSource = vtk.vtkLineSource()
lineSource.SetPoint1(-ld, 0.0, 0.0)
lineSource.SetPoint2(lu, 0.0, 0.0)
particleCSource = vtk.vtkTubeFilter()
particleCSource.SetInputConnection(lineSource.GetOutputPort())
particleCSource.SetRadius(R)
particleCSource.SetNumberOfSides(10)
# particleCSource.CappingOn()

particlesC.SetSourceConnection(particleCSource.GetOutputPort())
particlesC.SetInputData(particleCPolys)
particlesCMapper = vtk.vtkPolyDataMapper()
particlesCMapper.SetInputConnection(particlesC.GetOutputPort())
particlesCActor = vtk.vtkActor()
particlesCActor.SetMapper(particlesCMapper)
ren.AddActor(particlesCActor)

particleESource = vtk.vtkSphereSource()
particleESource.SetRadius(0.95*R)
particleESource.SetThetaResolution(10)
particleESource.SetPhiResolution(10)

particleE1Points = vtk.vtkPoints()
particleE1Polys = vtk.vtkPolyData()
particleE1Polys.SetPoints(particleE1Points)
particlesE1 = vtk.vtkGlyph3D()
particlesE1.SetSourceConnection(particleESource.GetOutputPort())
particlesE1.SetInputData(particleE1Polys)
particlesE1Mapper = vtk.vtkPolyDataMapper()
particlesE1Mapper.SetInputConnection(particlesE1.GetOutputPort())
particlesE1Actor = vtk.vtkActor()
particlesE1Actor.SetMapper(particlesE1Mapper)
particlesE1Actor.GetProperty().SetColor(1, 0, 0)
ren.AddActor(particlesE1Actor)

particleE2Points = vtk.vtkPoints()
particleE2Polys = vtk.vtkPolyData()
particleE2Polys.SetPoints(particleE2Points)
particlesE2 = vtk.vtkGlyph3D()
particlesE2.SetSourceConnection(particleESource.GetOutputPort())
particlesE2.SetInputData(particleE2Polys)
particlesE2Mapper = vtk.vtkPolyDataMapper()
particlesE2Mapper.SetInputConnection(particlesE2.GetOutputPort())
particlesE2Actor = vtk.vtkActor()
particlesE2Actor.SetMapper(particlesE2Mapper)
ren.AddActor(particlesE2Actor)

first = True
for fname in args.dyns:
    dyn = np.load(fname.strip())

    try:
        r = pad_to_3d(dyn['r'])
        u = pad_to_3d(dyn['u'])
    except KeyError:
        print('Invalid dyn file %s' % fname)
        continue
    else:
        if 'R' in stat:
            in_slice = np.abs(r[:, -1]) / R_drop < args.cross
            r = r[in_slice]
            u = u[in_slice]

    particleCPoints.SetData(numpy_support.numpy_to_vtk(r))
    particleCPolys.GetPointData().SetVectors(numpy_support.numpy_to_vtk(u))

    re1 = r + u * lu
    particleE1Points.SetData(numpy_support.numpy_to_vtk(re1))

    re2 = r - u * ld
    particleE2Points.SetData(numpy_support.numpy_to_vtk(re2))

    timeActor.SetInput(fname)

    renWin.Render()
    renWin.SetWindowName(fname)

    if first:
        # ren.GetActiveCamera().Zoom(1.5)
        first = False
    if args.save:
        print(fname)
        fname = os.path.splitext(os.path.basename(fname))[0]
        winImFilt.Modified()
        writer.SetFileName('%s.png' % fname)
        writer.Write()
    elif multis:
        time.sleep(dt)
if args.save:
    if multis:
        writer.End()
else:
    iren.Start()
