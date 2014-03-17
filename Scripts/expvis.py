#!/usr/bin/env python

import os
import time
import argparse
import numpy as np
import vtk
from vtk.util import numpy_support
import utils
import butils
import droplyse

parser = argparse.ArgumentParser(description='Visualise system states using VTK')
parser.add_argument('csv', 
    help='xyz csv file')
parser.add_argument('-s', '--save', default=False, action='store_true',
    help='Save plot')
parser.add_argument('-c', '--cross', type=float, default=1.0,
    help='Cross section fraction')
args = parser.parse_args()

ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.SetSize(600, 600)
renWin.AddRenderer(ren)

R = 0.551

xyz, n, R_drop, hemisphere = droplyse.parse(args.csv)

envMapper = vtk.vtkPolyDataMapper()
envActor = vtk.vtkActor()
env = vtk.vtkSphereSource()
env.SetThetaResolution(30)
env.SetPhiResolution(30)
env.SetRadius(R_drop)
envMapper.SetInputConnection(env.GetOutputPort())
envActor.SetMapper(envMapper)
envActor.GetProperty().SetColor(0, 1, 0)
envActor.GetProperty().SetRepresentationToWireframe()
envActor.GetProperty().SetOpacity(0.5)
ren.AddActor(envActor)

particlePoints = vtk.vtkPoints()
particlePolys = vtk.vtkPolyData()
particlePolys.SetPoints(particlePoints)
particles = vtk.vtkGlyph3D()
particleSource = vtk.vtkSphereSource()
particleSource.SetRadius(R)
particleSource.SetThetaResolution(10)
particleSource.SetPhiResolution(10)
particles.SetSourceConnection(particleSource.GetOutputPort())
particles.SetInputData(particlePolys)
particlesMapper = vtk.vtkPolyDataMapper()
particlesMapper.SetInputConnection(particles.GetOutputPort())
particlesActor = vtk.vtkActor()
particlesActor.SetMapper(particlesMapper)
particlesActor.GetProperty().SetColor(1, 0, 0)
ren.AddActor(particlesActor)

in_slice = np.abs(xyz[:, -1]) / R_drop < args.cross
xyz = xyz[in_slice]

xyz = np.ascontiguousarray(xyz)

particlePoints.SetData(numpy_support.numpy_to_vtk(xyz))

if args.save:
    renWin.OffScreenRenderingOn()
    winImFilt = vtk.vtkWindowToImageFilter()
    winImFilt.SetInput(renWin)
    writer = vtk.vtkPNGWriter()
    writer.SetInputConnection(winImFilt.GetOutputPort())
    fname = os.path.splitext(os.path.basename(args.csv))[0]
    winImFilt.Modified()
    writer.SetFileName('%s.png' % fname)
    writer.Write()
else:
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.Initialize()
    iren.Start()
