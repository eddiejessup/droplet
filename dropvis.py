#!/usr/bin/env python

import os
import argparse
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk


def progress_renwin(renWin):
    iren = renWin.GetInteractor()
    key = iren.GetKeySym()

    if key == 'Right':
        renWin.index += 1
    elif key == 'Left':
        renWin.index -= 1
    fname = renWin.fnames[renWin.index]

    dyn = np.load(fname.strip())

    r = dyn['r']
    u = dyn['u']

    if renWin.cross:
        in_slice = np.abs(r[:, -1]) < renWin.cross
        r = r[in_slice]
        u = u[in_slice]

    renWin.timeActor.SetInput(fname)

    renWin.particleCPoints.SetData(numpy_to_vtk(r))
    renWin.particleCPolys.GetPointData().SetVectors(numpy_to_vtk(u))

    re1 = r + u * renWin.l / 2.0
    renWin.particleE1Points.SetData(numpy_to_vtk(re1))

    re2 = r - u * renWin.l / 2.0
    renWin.particleE2Points.SetData(numpy_to_vtk(re2))

    renWin.Render()
    return fname


def progress_iren(obj, *args, **kwargs):
    progress_renwin(obj.GetRenderWindow())


def vis(dyns, save, cross):
    datdir = os.path.abspath(os.path.join(dyns[0], '../..'))
    stat = np.load('%s/static.npz' % datdir)
    l = stat['l']
    lu, ld = l / 2.0, l / 2.0
    R = stat['R']
    R_drop = stat['R_d']

    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.SetSize(600, 600)
    renWin.AddRenderer(ren)
    if save:
        renWin.OffScreenRenderingOn()
        winImFilt = vtk.vtkWindowToImageFilter()
        winImFilt.SetInput(renWin)
        writer = vtk.vtkPNGWriter()
        writer.SetInputConnection(winImFilt.GetOutputPort())
    else:
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        iren.Initialize()

    timeActor = vtk.vtkTextActor()
    timeActor.SetInput('init')
    ren.AddActor(timeActor)

    env = vtk.vtkSphereSource()
    env.SetThetaResolution(30)
    env.SetPhiResolution(30)
    env.SetRadius(R_drop)

    # env = vtk.vtkRegularPolygonSource()
    # env.GeneratePolygonOff()
    # env.SetNumberOfSides(200)
    # x = 4.0
    # th = np.arcsin(x / R_drop)
    # env.SetRadius(R_drop * np.cos(th))
    # env_tube = vtk.vtkTubeFilter()
    # env_tube.SetInputConnection(env.GetOutputPort())
    # env_tube.SetRadius(0.5)
    # env.Update()

    envMapper = vtk.vtkPolyDataMapper()
    envMapper.SetInputConnection(env.GetOutputPort())
    envActor = vtk.vtkActor()
    envActor.SetMapper(envMapper)
    envActor.GetProperty().SetColor(1, 0, 0)
    envActor.GetProperty().SetRepresentationToWireframe()
    envActor.GetProperty().SetOpacity(0.5)
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
    particlesC.SetSourceConnection(particleCSource.GetOutputPort())
    particlesC.SetInputData(particleCPolys)
    particlesCMapper = vtk.vtkPolyDataMapper()
    particlesCMapper.SetInputConnection(particlesC.GetOutputPort())
    particlesCActor = vtk.vtkActor()
    particlesCActor.SetMapper(particlesCMapper)
    particlesCActor.GetProperty().SetColor(0, 1, 0)
    ren.AddActor(particlesCActor)
    particleESource = vtk.vtkSphereSource()
    particleESource.SetRadius(0.95 * R)
    particleESource.SetThetaResolution(20)
    particleESource.SetPhiResolution(20)
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
    particlesE1Actor.GetProperty().SetColor(0, 1, 0)
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
    particlesE2Actor.GetProperty().SetColor(0, 1, 0)
    ren.AddActor(particlesE2Actor)

    renWin.fnames = dyns
    renWin.index = 0
    renWin.l = l
    renWin.cross = cross
    renWin.timeActor = timeActor
    renWin.particleCPoints = particleCPoints
    renWin.particleCPolys = particleCPolys
    renWin.particleE1Points = particleE1Points
    renWin.particleE2Points = particleE2Points

    if not save:
        ren.GetActiveCamera().SetPosition(0.0, 0.0, -73.0)
        ren.GetActiveCamera().Zoom(1.0)

        iren.RemoveObservers('KeyPressEvent')
        iren.AddObserver('KeyPressEvent', progress_iren, 1.0)
        iren.Start()
    else:
        while True:
            fname = progress_renwin(renWin)
            print(fname)
            outname = os.path.splitext(os.path.basename(fname))[0]
            winImFilt.Modified()
            writer.SetFileName('{}.jpg'.format(outname))
            writer.Write()

parser = argparse.ArgumentParser(
    description='Visualise system states using VTK')
parser.add_argument('dyns', nargs='*',
                    help='npz files containing dynamic states')
parser.add_argument('-s', '--save', default=False, action='store_true',
                    help='Save plot')
parser.add_argument('-c', '--cross', type=float, default=None,
                    help='Cross section fraction')
args = parser.parse_args()

vis(args.dyns, args.save, args.cross)
