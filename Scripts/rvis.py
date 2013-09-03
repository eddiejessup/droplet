#! /usr/bin/env python

from __future__ import print_function
import os
import time
import argparse
import numpy as np
import vtk
from vtk.util import numpy_support
import butils

def pad_to_3d(a):
    return np.pad(a, ((0, 0), (0, 3 - a.shape[1])), mode='constant')

parser = argparse.ArgumentParser(description='Visualise system states using VTK')
parser.add_argument('dyns', nargs='*',
    help='npz files containing dynamic states')
parser.add_argument('-s', '--save', default=False, action='store_true',
    help='Save plot')
args = parser.parse_args()

multis = len(args.dyns) > 1
datdir = os.path.abspath(os.path.join(args.dyns[0], '../..'))

stat = butils.get_stat(datdir)
L = stat['L']

if multis:
    dt = butils.t(args.dyns[1]) - butils.t(args.dyns[0])

# Create render window
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
# renWin.SetSize(600, 600)
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
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.Initialize()

# System bounds
env = vtk.vtkCubeSource()
env.SetXLength(L)
env.SetYLength(L)
env.SetZLength(L)
envMapper = vtk.vtkPolyDataMapper()
envMapper.SetInputConnection(env.GetOutputPort())
envActor = vtk.vtkActor()
envActor.GetProperty().SetOpacity(0.2)
envActor.SetMapper(envMapper)
ren.AddActor(envActor)

# Time
timeActor = vtk.vtkTextActor()
ren.AddActor(timeActor)

# Obstructions
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

    obs = vtk.vtkGlyph3D()
    obs.SetSourceConnection(cubeSource.GetOutputPort())
    obs.SetInputData(polypoints)
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

    obs = vtk.vtkGlyph3D()
    obs.SetSourceConnection(sphereSource.GetOutputPort())
    obs.SetInputData(polypoints)
elif 'R' in stat:
    R_drop = stat['R']

    obs = vtk.vtkSphereSource()
    obs.SetThetaResolution(30)
    obs.SetPhiResolution(30)
    obs.SetRadius(R_drop)
if 'obs' in locals():
    obsMapper = vtk.vtkPolyDataMapper()
    obsMapper.SetInputConnection(obs.GetOutputPort())
    obsActor = vtk.vtkActor()
    obsActor.SetMapper(obsMapper)
    obsActor.GetProperty().SetColor(1, 0, 0)
    obsActor.GetProperty().SetOpacity(0.2)
    obsActor.GetProperty().SetRepresentationToWireframe()
    ren.AddActor(obsActor)

# Particles
r_0 = pad_to_3d(stat['r_0'])
particlePoints = vtk.vtkPoints()
particlePolys = vtk.vtkPolyData()
particlePolys.SetPoints(particlePoints)
particle_scale = 0.02 * L
scales_npy = np.ones([len(r_0)]) * particle_scale
scales = numpy_support.numpy_to_vtk(scales_npy)
particlePolys.GetPointData().SetScalars(scales)
particles = vtk.vtkGlyph3D()
particleSource = vtk.vtkArrowSource()
particleSource.SetTipRadius(0.2)
particleSource.SetTipLength(1.0)
particleSource.SetTipResolution(5)
particleSource.SetShaftRadius(0.0)
particles.SetSourceConnection(particleSource.GetOutputPort())
particles.SetInputData(particlePolys)
particlesMapper = vtk.vtkPolyDataMapper()
particlesMapper.SetInputConnection(particles.GetOutputPort())
particlesActor = vtk.vtkActor()
particlesActor.SetMapper(particlesMapper)
ren.AddActor(particlesActor)

for fname in args.dyns:
    # Get state
    dyn = np.load(fname.strip())

    # Get data
    try:
        r = pad_to_3d(dyn['r'])
        v = pad_to_3d(dyn['v'])
    except KeyError:
        print('Invalid dyn file %s' % fname)
        continue

    # Update actors
    particlePoints.SetData(numpy_support.numpy_to_vtk(r))
    particlePolys.GetPointData().SetVectors(numpy_support.numpy_to_vtk(v))
    timeActor.SetInput(str(butils.t(fname)))

    # Update plot
    renWin.Render()

    # Save if necessary
    if args.save:
        print(fname)
        fname = os.path.splitext(os.path.basename(fname))[0]
        winImFilt.Modified()
        if not multis:
            writer.SetFileName('%s.png' % fname)
        writer.Write()
    # elif multis:
    #     time.sleep(dt)

if args.save:
    if multis:
        writer.End()
else:
    iren.Start()
