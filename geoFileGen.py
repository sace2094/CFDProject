"""Code: MCEN 4231/5231 Assignment Supplement
Author: Debanjan Mukherjee (instructor)
Institution: University of Colorado Boulder
Last Revised: 2024

"""

import numpy as np
import sys
import os
import math

def writeGeoFromXYZVarySizing(a_XYZFile, a_GeoFile, Inputs, a_ParametricSize=True):
    """Function to compose a Gmsh compatible '*.geo' file to include
    all the points from a specified file with formatted point coordinates

    Args:
        a_XYZFile: file with formatted point coordinates (space separated)
        a_GeoFile: output geo file where all points are to be loaded
        Inputs: A file with inputs that are needed for scaling points and mesh
        a_ParametricSize: set this to True if you want to leave size as a parameter h

    """
    with open(Inputs) as inputFile:
        line = inputFile.readline()
        linesplit = line.split()
        Chord = float(linesplit[0])
        Height = float(linesplit[1])
        AOA = float(linesplit[2])
    Theta = math.radians(AOA)



    geoFile = open(a_GeoFile,"w+")

    with open(a_XYZFile) as xyzFile:
        Refinement = """E1 = 0.02;
            E2 = 0.01; 
            E3 = 1.0;
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
    Points = f"Point(100) = {{-3, -3, 0, {Chord}}}; " \
         f"Point(101) = {{7, -3, 0, {Chord}}}; " \
         f"Point(102) = {{7, 3, 0, {Chord}}}; " \
         f"Point(103) = {{-3, 3, 0, {Chord}}};\n"
    Lines = """Line(101) = {100,101}; 
            Line(102) = {101,102}; 
            Line(103) = {102,103}; 
            Line(104) = {103,100}; """

    Curves = """Line Loop(20) = {101,102,103,104};

        Plane Surface(1) = {20,10};

        Physical Surface("fluid") = {1};

        Physical Curve("airfoil") = {1};

        Physical Curve("inlet") = {104};
        Physical Curve("outlet") = {102};
        Physical Curve("top") = {103};
        Physical Curve("bottom") = {101};
        """
    Fields = """Field[1] = Distance;
            Field[1].CurvesList = {1}; // your airfoil curves

            Field[2] = Threshold;
        Field[2].IField = 1;

        Field[2].LcMin = E1;
        Field[2].LcMax = E3;

        Field[2].DistMin = 0.0;
        Field[2].DistMax = 0.5;

        Field[3] = Box;
        Field[3].VIn  = E2;
        Field[3].VOut = E3;

        Field[3].XMin = -0.1;
        Field[3].XMax =  0.1;
        Field[3].YMin = -0.1;
        Field[3].YMax =  0.1;

        Field[4] = Box;
        Field[4].VIn  = E2;
        Field[4].VOut = E3;

        Field[4].XMin = 0.9;
        Field[4].XMax = 1.1;
        Field[4].YMin = -0.1;
        Field[4].YMax = 0.1;

        Field[5] = Box;
        Field[5].VIn  = E3;
        Field[5].VOut = E3;

        Field[5].XMin = -3;
        Field[5].XMax =  7;
        Field[5].YMin = -3;
        Field[5].YMax =  3;

        Field[6] = Min;
        Field[6].FieldsList = {2,3,4,5};

        Background Field = 6;"""

    geoFile.write(Spline + Line + Points + Lines + Curves + Fields)
    geoFile.close()


if __name__ == "__main__":
    """This runs the script.
    Users have to modify the following variables
    xyzFileName:  enter the filename with the coordinates
    geoFileName:  enter the filename of the Gmsh geo file

    """

    if len(sys.argv) != 4:
        print('The program needs to be called as follows:')
        print('python3 geoWriter.py xyzFile geoFile inputFile')
        sys.exit()
    else:
        xyzFileName = sys.argv[1]
        geoFileName = sys.argv[2]
        Inputs = sys.argv[3]
        writeGeoFromXYZVarySizing(xyzFileName, geoFileName, Inputs)
