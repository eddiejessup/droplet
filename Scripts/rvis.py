#!/usr/bin/env python

import os
import time
import argparse
import time
import numpy as np
import vtk
from vtk.util import numpy_support

def pad_to_3d(a):
    a_pad = np.zeros([len(a), 3], dtype=a.dtype)
    a_pad[:, :a.shape[-1]] = a
    return a_pad

parser = argparse.ArgumentParser(description='Visualise system states using VTK')
parser.add_argument('static',
    help='npz file containing static state')
parser.add_argument('dyns', nargs='*',
    help='npz files containing dynamic states')
parser.add_argument('-s', '--save', default=False, action='store_true',
    help='Save plot')
args = parser.parse_args()

# create a rendering window and renderer
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.SetSize(600, 600)
renWin.AddRenderer(ren)
# create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
if args.save:
    # create a window to image filter attached to render window
    winImFilt = vtk.vtkWindowToImageFilter()
    winImFilt.SetInput(renWin)
    writer = vtk.vtkPNGWriter()
    writer.SetInputConnection(winImFilt.GetOutputPort())

stat = np.load(args.static)

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
sysActor.GetProperty().SetOpacity(0.5)
sysActor.SetMapper(sysMapper)
# ren.AddActor(sysActor)

# Obstacles
# Mapper
envMapper = vtk.vtkPolyDataMapper()
envActor = vtk.vtkActor()

if 'o' in stat:
    o = stat['o']

    dx = L / o.shape[0]
    inds = np.indices(o.shape).reshape((o.ndim,-1)).T
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
    r = stat['r']
    R = stat['R']

    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(r))
    polypoints = vtk.vtkPolyData()
    polypoints.SetPoints(points)

    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetThetaResolution(30)
    sphereSource.SetPhiResolution(30)
    sphereSource.SetRadius(R)

    env = vtk.vtkGlyph3D()
    env.SetSourceConnection(sphereSource.GetOutputPort())
    env.SetInputData(polypoints)

elif 'R' in stat:
    R = stat['R']

    env = vtk.vtkSphereSource()
    env.SetThetaResolution(30)
    env.SetPhiResolution(30)
    env.SetRadius(R)

envMapper.SetInputConnection(env.GetOutputPort())
envActor.SetMapper(envMapper)
envActor.GetProperty().SetColor(1, 0, 0)
envActor.GetProperty().SetOpacity(0.3)
# ren.AddActor(envActor)

# Particles
# Poly
r_0 = pad_to_3d(stat['r_0'])
particlePoints = vtk.vtkPoints()
particlePoints.SetData(numpy_support.numpy_to_vtk(r_0))
particlePolys = vtk.vtkPolyData()
particlePolys.SetPoints(particlePoints)

sphereSource = vtk.vtkSphereSource()
sphereSource.SetThetaResolution(4)
sphereSource.SetPhiResolution(4)
sphereSource.SetRadius(0.5)

particles = vtk.vtkGlyph3D()
particles.SetSourceConnection(sphereSource.GetOutputPort())
particles.SetInputData(particlePolys)

# mapper
particlesMapper = vtk.vtkPolyDataMapper()
particlesMapper.SetInputConnection(particles.GetOutputPort())
# actor
particlesActor = vtk.vtkActor()
particlesActor.SetMapper(particlesMapper)
particlesActor.GetProperty().SetColor(0, 1, 0)
envActor.GetProperty().SetOpacity(0.8)
ren.AddActor(particlesActor)

iren.Initialize()

for fname in args.dyns:
    dyn = np.load(fname.strip())
    r = pad_to_3d(dyn['r'])
    particlePoints.SetData(numpy_support.numpy_to_vtk(r))

    renWin.Render()
    # ren.GetActiveCamera().Azimuth(1)

    if args.save:
        fname = os.path.splitext(os.path.basename(fname))[0]
        winImFilt.Modified()
        writer.SetFileName('img/%s.png' % fname)
        writer.Write()
    else:
        time.sleep(0.1)
iren.Start()
    