import basix
from mpi4py import MPI
import dolfinx as dfx
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem.petsc import NewtonSolverNonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import dolfinx.io.gmsh as gmshio
from petsc4py import PETSc
import ufl as ufl
import matplotlib.pyplot as plt
import numpy as np
import time
import gmsh
import sys
import os
import math





def writeGeoFromXYZVarySizing(a_XYZFile, Inputs, a_ParametricSize=True):
    """Function to compose a Gmsh compatible '*.geo' file to include
    all the points from a specified file with formatted point coordinates

    Args:
        a_XYZFile: file with formatted point coordinates (space separated)
        a_GeoFile: output geo file where all points are to be loaded
        Inputs: A file with inputs that are needed for scaling points and mesh
        a_ParametricSize: set this to True if you want to leave size as a parameter h

    """
    temp = os.path.splitext(a_XYZFile)[0]

    geofile = f"{temp}.geo"
    meshfile = f"{temp}.msh"

    Chord = Inputs[0]
    Height = Inputs[1]
    AOA = Inputs[2]
    BoxH = Inputs[3]
    BoxL = Inputs[4]

    ymax = BoxH 
    ymin = 0
    xmax = 2 * (BoxL/3)
    xmin = -(BoxL/3)
    Theta = math.radians(AOA)



    with open(geofile,"w+") as geoFile:

        with open(a_XYZFile) as xyzFile:
            Refinement = """
                h = 1;"""
            geoFile.write(Refinement)

            next(xyzFile)

            for i, line in enumerate(xyzFile, start=1):

                lineDict = line.split()

                x = float(lineDict[0])
                y = float(lineDict[1])
                z = float(lineDict[2])

                x = x-0.25

                x = x * Chord
                y = y*Chord

                y = -y

                x_new = (x * math.cos(Theta) - y * math.sin(Theta))
                y_new = (x * math.sin(Theta) + y * math.cos(Theta)) + Height

                x_new += 0.25 * Chord

                if a_ParametricSize == True:
                    geoLine = f"Point({i}) = {{{x_new},{y_new},{z},h}};\n"
                else:
                    h = lineDict[3]
                    geoLine = f"Point({i}) = {{{x_new},{y_new},{z},{h}}};\n"

                geoFile.write(geoLine)
    
        Spline = f"Spline(1) = {{1:{i}, 1}};\n"
        Line = f"Line Loop(10) = {{-1}};\n"

        Points = f"Point(100) = {{{xmin}, {ymin}, 0, {Chord}}}; " \
             f"Point(101) = {{{xmax}, {ymin}, 0, {Chord}}}; " \
             f"Point(102) = {{{xmax}, {ymax}, 0, {Chord}}}; " \
             f"Point(103) = {{{xmin}, {ymax}, 0, {Chord}}};\n"
        
        Lines = """Line(101) = {100,101}; 
                Line(102) = {101,102}; 
                Line(103) = {102,103}; 
                Line(104) = {103,100}; """

        Curves = """Line Loop(20) = {101,102,103,104};

            Plane Surface(1) = {20,10};

            Physical Surface("fluid") = {1};

            Physical Curve(1) = {1};      // airfoil
            Physical Curve(2) = {104};    // inlet
            Physical Curve(3) = {102};    // outlet
            Physical Curve(4) = {103};    // top
            Physical Curve(5) = {101};    // bottom
            """
        Fields = f"""
                        Chord = {Chord};

                        E1 = 0.001 * Chord;  // near airfoil
                        E2 = 0.05  * Chord;  // wake refinement
                        E3 = 0.4   * Chord;  // far-field

                        Field[1] = Distance;
                        Field[1].CurvesList = {{1}};
                        Field[1].Sampling = 100;

                        Field[2] = Threshold;
                        Field[2].IField = 1;
                        Field[2].LcMin = E1;
                        Field[2].LcMax = E3;
                        Field[2].DistMin = 0.01 * Chord;
                        Field[2].DistMax = 0.3 * Chord;

                        
                        Field[3] = Box;
                        Field[3].VIn  = E2;
                        Field[3].VOut = E3;
                        Field[3].XMin = {0 - 0.75 * Chord};
                        Field[3].XMax = {xmax};
                        Field[3].YMin = {Height - 1.25 * Chord};
                        Field[3].YMax = {Height + 1.75 * Chord};

                        
                        Field[4] = Box;
                        Field[4].VIn  = E1;
                        Field[4].VOut = E3;
                        Field[4].XMin = {-0.2 * Chord};
                        Field[4].XMax = {1.25 * Chord};
                        Field[4].YMin = {Height - .5 * Chord};
                        Field[4].YMax = {Height + .5 * Chord};

                        Field[5] = BoundaryLayer;
                        Field[5].CurvesList = {{1}};
                        Field[5].Size = 0.001 * Chord;
                        Field[5].Ratio = 1.15;
                        Field[5].Thickness = 0.03 * Chord;

                        Field[6] = Min;
                        Field[6].FieldsList = {{2, 3, 4, 5}};
                    
                        Background Field = 6;
                        """

        geoFile.write(Spline + Line + Points + Lines + Curves + Fields)

    geoFile.close()

    gmsh.initialize()
    gmsh.open(geofile)

    gmsh.model.mesh.generate(2)
    nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes()
    num_nodes = len(nodeTags)

    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements()

    num_2d_elements = 0
    for etype, tags in zip(elemTypes, elemTags):
        name, dim, order, numv, localCoords, _ = gmsh.model.mesh.getElementProperties(etype)
    if dim == 2:
        num_2d_elements += len(tags)

    gmsh.write(meshfile)

    gmsh.finalize()

    print(f"\nGenerated files:")
    print(f"  {geofile}")
    print(f"  {meshfile}\n")
    print(f"Mesh contains {num_nodes} nodes and {num_2d_elements} elements.")

    return meshfile




height = 1 #how high from ground (will likely get ranged)
AOA = 0 # degrees, also will get ranged
chord = 1 #scales chord of airfoil, leave at one for now
boxH =  9* chord #sets height of bounding box for sim
boxL = 15 * chord #sets length of bounding box for sim


Inputs = [chord, height, AOA, boxH, boxL]
xyzFileName = sys.argv[1]
meshFile = writeGeoFromXYZVarySizing(xyzFileName, Inputs)

