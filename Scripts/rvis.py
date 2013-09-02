#!/usr/bin/env python

import os
import time
import argparse
import time
import numpy as np
import vtk
from vtk.util import numpy_support
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

    if multis:
        writer = vtk.vtkOggTheoraWriter()
        writer.SetFileName('out.ogv')
        writer.SetRate(1.0 / dt)
        writer.SetQuality(0)
    else:
        writer = vtk.vtkPNGWriter()
    writer.SetInputConnection(winImFilt.GetOutputPort())
    if multis:
        writer.Start()
else:
    # create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.Initialize()

stat = butils.get_stat(datdir)

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
# Particles
# Poly
r_0 = pad_to_3d(stat['r_0'])
particlePoints = vtk.vtkPoints()
particlePolys = vtk.vtkPolyData()
particlePolys.SetPoints(particlePoints)
# can't use numpy to vtk conversion funnction here for some reason
particle_scale = 0.02
particle_scale *= L
scales = vtk.vtkFloatArray()
for i in range(len(r_0)):
    scales.InsertNextValue(particle_scale)
scales.SetName("scales")
particlePolys.GetPointData().SetScalars(scales)

particles = vtk.vtkGlyph3D()

particleSource = vtk.vtkArrowSource()
particleSource.SetTipRadius(0.2)
particleSource.SetTipLength(1.0)
particleSource.SetTipResolution(10)
particleSource.SetShaftRadius(0.0)

particles.SetSourceConnection(particleSource.GetOutputPort())
particles.SetInputData(particlePolys)

# mapper
particlesMapper = vtk.vtkPolyDataMapper()
particlesMapper.SetInputConnection(particles.GetOutputPort())
# actor
particlesActor = vtk.vtkActor()
particlesActor.SetMapper(particlesMapper)
particlesActor.GetProperty().SetColor(0, 1, 0)
# particlesActor.GetProperty().SetOpacity(0.8)
ren.AddActor(particlesActor)

particlePoints.SetData(numpy_support.numpy_to_vtk(r_0))
renWin.Render()
ren.GetActiveCamera().Zoom(2.0)
ren.GetActiveCamera().Azimuth(10.0)
    env.SetRadius(R_drop)

try:
    envMapper.SetInputConnection(env.GetOutputPort())
except NameError:
    pass
else:
    envActor.SetMapper(envMapper)
    envActor.GetProperty().SetColor(1, 0, 0)
    envActor.GetProperty().SetOpacity(0.2)
    envActor.GetProperty().SetRepresentationToWireframe()
    ren.AddActor(envActor)


first = True
for fname in args.dyns:
    dyn = np.load(fname.strip())
    try:
        r = pad_to_3d(dyn['r'])
        v = pad_to_3d(dyn['v'])
    except KeyError:
        print('Invalid dyn file %s' % fname)
        continue
    particlePoints.SetData(numpy_support.numpy_to_vtk(r))
    particlePolys.GetPointData().SetVectors(numpy_support.numpy_to_vtk(v))

    renWin.Render()
    renWin.SetWindowName(fname)

    if first:
        ren.GetActiveCamera().Azimuth(20.0)
        # ren.GetActiveCamera().Yaw(1.0)
        # ren.GetActiveCamera().Pitch(1.0)
        ren.GetActiveCamera().Zoom(1.5)
        first = False
    # ren.GetActiveCamera().Azimuth(0.5)
    # ren.GetActiveCamera().Zoom(1.01)
    if args.save:
        print(fname)
        fname = os.path.splitext(os.path.basename(fname))[0]
        winImFilt.Modified()
        if not multis:
            writer.SetFileName('%s.png' % fname)
        writer.Write()
    elif multis:
        time.sleep(5.0*dt)
if args.save:
    if multis:
        writer.End()
else:
    iren.Start()
