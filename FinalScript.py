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
import math

#--------------------------------------------------------------------------
# function defined to post-process the force on the cylinder by integrating
# the total fluid stress on the surface of the cylinder
#--------------------------------------------------------------------------
def integrateFuidStress(a_U, a_P, a_Mu, a_N, a_Mesh, a_GammaP):

    eps = 0.5*(ufl.grad(a_U) + ufl.grad(a_U).T)
    sig = -a_P*ufl.Identity(2) + 2.0*a_Mu*eps

    traction = ufl.dot(sig, a_N)

    forceX = traction[0] * a_GammaP
    forceY = traction[1] * a_GammaP

    fXVal = dfx.fem.assemble_scalar(dfx.fem.form(forceX))
    fYVal = dfx.fem.assemble_scalar(dfx.fem.form(forceY))

    return [fXVal, fYVal]

#------------------------------------------------------------------------
# function defined to compute the eluted flux from the exit/outlet of the
# flow domain (computed as a function of time)
#------------------------------------------------------------------------
def integrateEluteFlux(a_C, a_U, a_D, a_N, a_Mesh, a_GammaP):

    flux = a_D * ufl.grad(a_C) - a_U * a_C
    c_f = ufl.dot(flux, a_N)
    elute = dfx.fem.assemble_scalar(dfx.fem.form(c_f * a_GammaP))

    return elute

def writeGeoFromXYZVarySizing(a_XYZFile, Inputs, a_ParametricSize=True):
    """Function to compose a Gmsh compatible '*.geo' file to include
    all the points from a specified file with formatted point coordinates

    Args:
        a_XYZFile: file with formatted point coordinates (space separated)
        a_GeoFile: output geo file where all points are to be loaded
        Inputs: A file with inputs that are needed for scaling points and mesh
        a_ParametricSize: set this to True if you want to leave size as a parameter h

    """
    meshfile = "MeshFile.msh"
    geofile = "geometry.geo"

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
            Refinement = """E1 = 0.005;
                E2 = 0.0025; 
                E3 = 0.5;
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

    gmsh.initialize()
    gmsh.open(geofile)

    gmsh.model.mesh.generate(2)
    gmsh.write(meshfile)

    gmsh.finalize()

    return meshfile

#------------------------------------------------------------------
# list of all file names where the output data will be written into
# - meshFile: reading mesh from an external file
# - outFileV: output file to write velocity data
# - outFileP: output file to write pressure data
# - outFileC: output file to write concentration data
# forceFile:
#------------------------------------------------------------------

#-----------------------------------------------------------
# list of all physics parameters that we have to specify
# - viscosity: dynamic viscosity of the fluid
# - density: fluid density
# - U0: parameter for setting inlet flow boundary condition
# - Dvalue: diffusivity of the species
# - Rvalue: any reaction constants
# - qvalue: Neumann flux specified at the cylinder
#-----------------------------------------------------------
viscosity = 0.002
density = 1.0
Ubar = 2.5
Dvalue = 0.01
Rvalue = 0.0
qvalue = 1.0
elemType = 'q1p1'


#---------------------------------------------------------------
# list of all geometry parameters that we have to specify
# - diam: diameter of the circular cylinder
# - boxH: height of the box domain (flow domain around cylinder)
# - boxL: length of the box domain (flow domain around cylinder)
#---------------------------------------------------------------
height = 3 #how high from ground (will likely get ranged)
AOA = 5 # degrees, also will get ranged
chord = 1 #scales chord of airfoil, leave at one for now
boxH = 8 * chord #sets height of bounding box for sim
boxL = 15 * chord #sets length of bounding box for sim


Inputs = [chord, height, AOA, boxH, boxL]
xyzFileName = sys.argv[1]
meshfile = writeGeoFromXYZVarySizing(xyzFileName, Inputs)

outFileV = 'V.xdmf'
outFileP = 'P.xdmf'
outFileC = 'C.xdmf'
forceFile = 'forces.csv'
fluxFile = 'flux.csv'
forceImage = 'forces.png'
eluteImage = 'elute.png'

#-------------------------------------------------------------------------------
# list of all time discretization parameters that we need to specify for this
# simulation:
# - dt: time step size (we will keep the same for NSE and ADR)
# - t_start: simulation start time
# - t_end: simulation end time
# - t_theta: the theta parameter in theta-Galerkin method
#-------------------------------------------------------------------------------
dt = 0.005
t_start = 0.0
t_end = 7.0
t_theta = 0.5

#-----------------------------------------------------------------------
# computing and reporting an estimate of the Reynolds number of the flow
#-----------------------------------------------------------------------
Reynolds = (Ubar * chord) / viscosity
print("The Problem Reynolds Number is:", Reynolds)

#---------------------------------------------------------------------------
# define now the functions that are needed to specify the different boundary
# conditions. note that we will need to define functions here for both the
# Navier-Stokes equations as well as the Advection-Diffusion equations
#---------------------------------------------------------------------------
def noSlipBC(x):
    return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))

def pressureBC(x):
    return np.zeros(x.shape[1])

def wallConcBC(x):
    return np.zeros(x.shape[1])

def inletBC(x):
    vals = np.zeros((mesh.geometry.dim, x.shape[1]))
    vals[0] = Ubar
    vals[1] = 0.0
    return vals
def movingWallBC(x):
    vals = np.zeros((mesh.geometry.dim, x.shape[1]))
    vals[0] = Ubar   # moving with freestream
    vals[1] = 0.0
    return vals

#--------------------------------------------------------------------------
# list out the IDs for each of the boundary segments, already marked in the
# Gmsh geo file using the Physical Curves and Physical Surfaces tags
# make sure that the IDs match exactly what was created in the geo file
#--------------------------------------------------------------------------
ID_AIRFOIL = 1
ID_INLET   = 2
ID_OUTLET  = 3
ID_TOP     = 4
ID_BOTTOM  = 5

#---------------------------------------------------------------------
# start logging the computational time using the time module in Python
#---------------------------------------------------------------------
startTime = time.time()

#-------------------------------------------------------------------------------
# loading the mesh
# here we will: (a) load the mesh, including the markers or IDs from an external
# mesh file; (b) retrieve the facet normals (that is n_hat for Gamma); and
# (c) save the topology dimensions
#-------------------------------------------------------------------------------
mesh_data = gmshio.read_from_msh(meshfile, comm=MPI.COMM_WORLD, gdim=2)
mesh = mesh_data.mesh
cell_markers = mesh_data.cell_tags
facet_markers = mesh_data.facet_tags

nVec = ufl.FacetNormal(mesh)

tdim = mesh.topology.dim
fdim = tdim - 1

#----------------------------------------------------------------------------
# create all the necessary vector function spaces for the problem
# we will not use a mixed function space of 3 variables as we are loosely
# coupling the system here, and will choose to solve the ADR equations for a
# concentration update after the NSE equations for velocity update are solved
#----------------------------------------------------------------------------
PE = basix.ufl.element('Lagrange', mesh.basix_cell(), degree=1)

if elemType == 'q2p1':
    QE = basix.ufl.element('Lagrange', mesh.basix_cell(), degree=2, shape=(mesh.topology.dim,))
elif elemType == 'q1p1':
    QE = basix.ufl.element('Lagrange', mesh.basix_cell(), degree=1, shape=(mesh.topology.dim,))

ME = basix.ufl.mixed_element([QE, PE])
W = dfx.fem.functionspace(mesh, ME)

C = dfx.fem.functionspace(mesh, ('Lagrange', 1))

#-----------------------------------------------------
# extract the submaps for the mixed function space for
# correct boundary condition assignment
#-----------------------------------------------------
U_sub, U_submap = W.sub(0).collapse()
P_sub, P_submap = W.sub(1).collapse()

#-------------------------------------------------------------------------------
# locate all the degrees of freedom : or Gamma_D for all boundaries; separately
# identified now for each of the physics equations/variables
# that is: degrees of freedom associated with same boundary but different
# physics variables must be stored differently
#-------------------------------------------------------------------------------
b_dofs_IN_F = dfx.fem.locate_dofs_topological((W.sub(0), U_sub), fdim, facet_markers.find(ID_INLET))
b_dofs_TOP_F = dfx.fem.locate_dofs_topological((W.sub(0), U_sub), fdim, facet_markers.find(ID_TOP))
b_dofs_OUT_F = dfx.fem.locate_dofs_topological((W.sub(1), P_sub), fdim, facet_markers.find(ID_OUTLET))
b_dofs_BOT_F = dfx.fem.locate_dofs_topological((W.sub(0), U_sub), fdim, facet_markers.find(ID_BOTTOM))
b_dofs_CYL_F = dfx.fem.locate_dofs_topological((W.sub(0), U_sub), fdim, facet_markers.find(ID_AIRFOIL))

b_dofs_IN_C = dfx.fem.locate_dofs_topological(C, fdim, facet_markers.find(ID_INLET))
b_dofs_TOP_C = dfx.fem.locate_dofs_topological(C, fdim, facet_markers.find(ID_TOP))
b_dofs_OUT_C = dfx.fem.locate_dofs_topological(C, fdim, facet_markers.find(ID_OUTLET))
b_dofs_BOT_C = dfx.fem.locate_dofs_topological(C, fdim, facet_markers.find(ID_BOTTOM))
b_dofs_CYL_C = dfx.fem.locate_dofs_topological(C, fdim, facet_markers.find(ID_AIRFOIL))

#-------------------------------------------------------------------------------
# obtain all the u_D definitions for the individual physics boundary conditions
# note: ensure that each boundary condition is associated with the correct
# physics equations and the associated correct function spaces
#-------------------------------------------------------------------------------
uD_Wall = dfx.fem.Function(U_sub)
uD_Wall.interpolate(noSlipBC)

uD_MovingWall = dfx.fem.Function(U_sub)
uD_MovingWall.interpolate(movingWallBC)

uD_Inlet = dfx.fem.Function(U_sub)
uD_Inlet.interpolate(inletBC)

uD_Outlet = dfx.fem.Function(P_sub)
uD_Outlet.interpolate(pressureBC)

uD_WallConc = dfx.fem.Function(C)
uD_WallConc.interpolate(wallConcBC)

#-------------------------------------------------------------------------------
# assign the boundary conditions and compile them
# recall: this is where we set:
# u = u_D for all x in Gamma_D
# for each Dirichlet boundary condition for each physics equations respectively
#-------------------------------------------------------------------------------
bc_INLET = dfx.fem.dirichletbc(uD_Inlet, b_dofs_IN_F, W.sub(0))
bc_TOP = dfx.fem.dirichletbc(uD_Wall, b_dofs_TOP_F, W.sub(0))
bc_OUTLET = dfx.fem.dirichletbc(uD_Outlet, b_dofs_OUT_F, W.sub(1))
bc_BOTTOM = dfx.fem.dirichletbc(uD_Wall, b_dofs_BOT_F, W.sub(0))
bc_CYL = dfx.fem.dirichletbc(uD_Wall, b_dofs_CYL_F, W.sub(0))

bc_IN_C = dfx.fem.dirichletbc(uD_WallConc, b_dofs_IN_C)
bc_TOP_C = dfx.fem.dirichletbc(uD_WallConc, b_dofs_TOP_C)
bc_BOT_C = dfx.fem.dirichletbc(uD_WallConc, b_dofs_BOT_C)

bc_NSE = [bc_INLET, bc_TOP, bc_OUTLET, bc_BOTTOM, bc_CYL]

bc_ADR = [bc_IN_C, bc_TOP_C, bc_BOT_C]

#----------------------------------------------------------------------------
# identify the boundaries associated with the Neumann conditions
# as well as any boundaries over which we will specifically compute any
# post-processed quantity.
# here: Gamma_CYL serves as the boundary over which we will compute the
# integrated fluid stresses and obtain estimates of flow-induced forces;
# however, Gamma_CYL for the ADR equations will also form the Neumann
# boundary from which a specified flux is specified
# likewise: the Gamma_OUT has to be defined such that we can actually compute
# the flux integral across the outlet end of the flow domain
#----------------------------------------------------------------------------
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_markers)
Gamma_CYL = ds(ID_AIRFOIL)
Gamma_OUT = ds(ID_OUTLET)

#---------------------------------------------------------------------
# define the trial and test functions based on the mixedfunctionspace
# for the NSE equations
# NOTE: the call to TrialFunctions and not TrialFunction etc.
# Likewise the call to TestFunctions and not TestFunction etc.
#---------------------------------------------------------------------
(v,q) = ufl.TestFunctions(W)

#----------------------------------------------------------------------------
# define the trialfunction and the testfunction objects for the ADR equations
# note: this is not a mixed space, hence the call to TrialFunction and not
# a call to TrialFunctions
#----------------------------------------------------------------------------
g = ufl.TestFunction(C)
c = ufl.TrialFunction(C)

#------------------------------------------------------------------
# defining all physics properties and discretization constants into
# standard dolfinx constant parameters
#------------------------------------------------------------------
mu = dfx.fem.Constant(mesh, dfx.default_scalar_type(viscosity))
rho = dfx.fem.Constant(mesh, dfx.default_scalar_type(density))
idt = dfx.fem.Constant(mesh, dfx.default_scalar_type(1.0/dt))
theta = dfx.fem.Constant(mesh, dfx.default_scalar_type(t_theta))
b = dfx.fem.Constant(mesh, PETSc.ScalarType((0.0,0.0)))
D = dfx.fem.Constant(mesh, dfx.default_scalar_type(Dvalue))
R = dfx.fem.Constant(mesh, dfx.default_scalar_type(Rvalue))
qf = dfx.fem.Constant(mesh, dfx.default_scalar_type(qvalue))

#-------------------------------------------------------------------------------
# Theta-Galerkin: for NSE problem:
#
# define the variational form terms without time derivative in current timestep
# in theta-Galerkin formulation this is corresponding to the t_n+1
#-------------------------------------------------------------------------------
W1 = dfx.fem.Function(W)
(u,p) = ufl.split(W1)

T1_1 = rho * ufl.inner(v, ufl.grad(u)*u) * ufl.dx
T2_1 = mu * ufl.inner(ufl.grad(v), ufl.grad(u)) * ufl.dx
T3_1 = p * ufl.div(v) * ufl.dx
T4_1 = q * ufl.div(u) * ufl.dx
T5_1 = rho * ufl.dot(v,b) * ufl.dx
L_1  = T1_1 + T2_1 - T3_1 -T4_1 - T5_1

#-------------------------------------------------------------------------------
# Theta-Galerkin: for NSE problem:
#
# define the variational form terms without time derivative in current timestep
# in theta-Galerkin formulation this is corresponding to the t_n
#-------------------------------------------------------------------------------
W0 = dfx.fem.Function(W)
(u0,p0) = ufl.split(W0)

T1_0 = rho * ufl.inner(v, ufl.grad(u0)*u0) * ufl.dx
T2_0 = mu * ufl.inner(ufl.grad(v), ufl.grad(u0)) * ufl.dx
T3_0 = p * ufl.div(v) * ufl.dx
T4_0 = q * ufl.div(u0) * ufl.dx
T5_0 = rho * ufl.dot(v,b) * ufl.dx
L_0 = T1_0 + T2_0 - T3_0 -T4_0 - T5_0

#--------------------------------------------------------------------------
# Theta-Galerkin: for NSE problem:
#
# combine variational forms with time derivative as discussed for the
# complete theta-Galerkin formulation with a one-step discretization of the
# time-derivative
#
#  dw/dt + L(t) = 0 is approximated as
#  (w-w0)/dt + (1-theta)*L(t0) + theta*L(t) = 0
#---------------------------------------------------------------------------
F_NSE = idt * rho * ufl.inner((u-u0),v) * ufl.dx + (1.0-theta) * L_0 + theta * L_1

#---------------------------------------------------------------------------
# Petrov-Galerkin Stabilization: for NSE problem:
#
# defining the stabilization parameter for the Petrov-Galerkin stabilization
#---------------------------------------------------------------------------
uNorm = ufl.sqrt(ufl.inner(u0, u0))
h = ufl.CellDiameter(mesh)
tau = ( (2.0*theta*idt)**2 + (2.0*uNorm/h)**2 + (4.0*mu/h**2)**2 )**(-0.5)

#----------------------------------------------------------------------
# Petrov-Galerkin Stabilization: for NSE problem:
#
# defining the complete residual for the Navier-Stokes momentum balance
#----------------------------------------------------------------------
residual = idt*rho*(u - u0) + \
    theta*(rho*ufl.grad(u)*u - mu*ufl.div(ufl.grad(u)) + ufl.grad(p) - rho*b) +\
    (1.0-theta)*(rho*ufl.grad(u0)*u0 - mu*ufl.div(ufl.grad(u0)) + ufl.grad(p) - rho*b)

#---------------------------------------------------------------------------
# Petrov-Galerkin Stabilization: for NSE problem:
#
# including now the contributions from the SUPG and PSPG stabilization terms
# into the overall theta-Galerkin weak form
#---------------------------------------------------------------------------
F_SUPG_NSE = tau * ufl.inner(ufl.grad(v)*u, residual) * ufl.dx

if elemType == 'q1p1':
    F_PSPG_NSE = - tau * ufl.inner(ufl.grad(q), residual) * ufl.dx

F_NSE = F_NSE + F_SUPG_NSE + F_PSPG_NSE

#-------------------------------------------------------------------
# Theta-Galerkin: for ADR problem:
#
# define the solutions c_n and c_n+1 - placeholders for solutions at
# time t_n and t_n+1 respectively
#-------------------------------------------------------------------
c0  = dfx.fem.Function(C)
c1  = dfx.fem.Function(C)

#---------------------------------------------------------------------
# Theta-Galerkin: for ADR problem:
#
# define the diffusion contribution to the weak form and matrix system
# evaluated now at t_n and t_n+1 respectively
#---------------------------------------------------------------------
K0 = D * ufl.inner(ufl.grad(g), ufl.grad(c0)) * ufl.dx
K1 = D * ufl.inner(ufl.grad(g), ufl.grad(c)) * ufl.dx

#--------------------------------------------------------------------------
# Theta-Galerkin: for ADR problem:
#
# define the advection contribution to the weak form and matrix system
# evaluated now at t_n and t_n+1 respectively
#
# IMPOTANT: this is where the two physics equations couple to each other
# specifically, the advection velocity in the ADR problem is to be obtained
# a solution of the velocity update in the NSE equation. here, we therefore
# take the known velocity solution at t_n == u0 to be the velocity used for
# the advection portion of the ADR
#--------------------------------------------------------------------------
U0 = ufl.inner(g, ufl.dot(u0, ufl.grad(c0))) * ufl.dx
U1 = ufl.inner(g, ufl.dot(u0, ufl.grad(c))) * ufl.dx

#-------------------------------------------------------------------------------
# Theta-Galerkin: for ADR problem:
# define the reaction contribution to the weak form and matrix system
# then
# extract the Neumann boundary portion for the domain
# by extracting out all the boundary facets into the object 'ds' based on the
# facet_marker information, we can now restrict the integral onto only the
# portion of the boundary with id = ID_Y0 by assembling the integral over the
# portion ds(ID_CYL)
#-------------------------------------------------------------------------------
fR = g * R * ufl.dx
fN = g * qf * Gamma_CYL

#--------------------------------------------------------------------------
# Theta-Galerkin: for ADR problem:
#
# combine variational forms with time derivative as discussed for the
# complete theta-Galerkin formulation with a one-step discretization of the
# time-derivative
#
#  dw/dt + L(t) = 0 is approximated as
#  (w-w0)/dt + (1-theta)*L(t0) + theta*L(t) = 0
#---------------------------------------------------------------------------
F_ADR = (1.0/dt) * ufl.inner(g, c) * ufl.dx - (1.0/dt) * ufl.inner(g,c0)*ufl.dx \
    + theta * K1 + theta * U1 \
    + (1.0 - theta) * K0 + (1.0 - theta) * U0 \
    - fN - fR

#---------------------------------------------------------------------------
# Petrov-Galerkin Stabilization: for ADR problem:
#
# defining the stabilization parameter for the Petrov-Galerkin stabilization
#---------------------------------------------------------------------------
uNorm = ufl.sqrt(ufl.inner(u0, u0))
h = ufl.CellDiameter(mesh)
tau = ( (2.0*uNorm/h)**2 + 9.0*(4.0*D/(h*h))**2 )**(-0.5)

#----------------------------------------------------------------------
# Petrov-Galerkin Stabilization: for ADR problem:
#
# defining the complete residual for the Advection Diffusion equation
#----------------------------------------------------------------------
residual = (1.0/dt)*(c1 - c0) + ufl.dot(u0, ufl.grad(c)) - ufl.div(D*ufl.grad(c))

#---------------------------------------------------------------------------
# Petrov-Galerkin Stabilization: for ADR problem:
#
# including now the contributions from the SUPG (only) stabilization terms
# into the overall theta-Galerkin weak form
#---------------------------------------------------------------------------
F_SUPG_ADR = tau * ufl.inner(ufl.dot(u0, ufl.grad(g)), residual) * ufl.dx

F_ADR = F_ADR + F_SUPG_ADR

#--------------------------------------------------------------------
# define and configure the details of a Newton Solver for the overall
# NonlinearProblem defined in F_NSE above, set as F_NSE = 0
# this is the non linear piece of the problem
#--------------------------------------------------------------------
problem_nse = NewtonSolverNonlinearProblem(F_NSE, W1, bcs=bc_NSE)
solver_nse = NewtonSolver(MPI.COMM_WORLD, problem_nse)
solver_nse.convergence_criterion = "incremental"
solver_nse.rtol = 1e-7
solver_nse.report = True

ksp = solver_nse.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "gmres"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "umfpack"
ksp.setFromOptions()

#---------------------------------------------------------------------
# mean while we can use the weak form for the ADR equation to
# define a linear solver; and assemble the global matrix vector system
# based on the combined Theta-Galerkin weak form F_ADR
#---------------------------------------------------------------------
mat_adr = ufl.lhs(F_ADR)
vec_adr = ufl.rhs(F_ADR)

problem_adr = LinearProblem(mat_adr, vec_adr, bcs=bc_ADR, u=c1, \
    petsc_options_prefix='adr',\
    petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "umfpack"})

t = t_start
tn = 0

#---------------------------------------------------------------------------
# configure all the output files
# the ASCII encoding in the xdmf outputs is needed for opening the files in
# Windows as well as Mac/Linux; also xdmf will not handle the cases where
# the polynomial order of the elements is higher than 1
#---------------------------------------------------------------------------
vFile = dfx.io.XDMFFile(mesh.comm, outFileV, "w", encoding=dfx.io.XDMFFile.Encoding.ASCII)
vFile.write_mesh(mesh)

pFile = dfx.io.XDMFFile(mesh.comm, outFileP, "w", encoding=dfx.io.XDMFFile.Encoding.ASCII)
pFile.write_mesh(mesh)

cFile = dfx.io.XDMFFile(mesh.comm, outFileC, "w", encoding=dfx.io.XDMFFile.Encoding.ASCII)
cFile.write_mesh(mesh)

#-------------------------------------------------------------------
# initialize a number of arrays/lists for storing the post-processed
# quantities from the simulation
#-------------------------------------------------------------------
time_Arr = []
fX_Arr = []
fY_Arr = []
cD_Arr = []
cL_Arr = []
elute_Arr = []

#-----------------------------------------------------------------------
# structure of the time loop:
# - inner loop: nonlinear solver convergence for NSE
# - swap solution from t_n+1 to t_n for velocity and pressure
# - this updates u0 from previous step to store the updated uf
# - use this 'new' u0 to solve for c1 - updated concentration
# - then compute post-processed forces using updated velocity, pressure
# - then compute post-processed flux using updated concentration
# - then write all updated output data into output files
# - this closes the outer loop
#----------------------------------------------------------------------

print('Starting loop:')

while t < t_end:

    t_in = time.time()
    n, converged = solver_nse.solve(W1)
    assert (converged)
    t_out = time.time()

    print(f"t = {t:.6f}; Number of iterations: {n:d}; compute time: {t_out-t_in:f}")

    uf = W1.split()[0].collapse()
    pf = W1.split()[1].collapse()

    uf.name = 'vel'
    pf.name = 'pres'

    W0.x.array[:] = W1.x.array

    c1 = problem_adr.solve()
    c0.x.array[:] = c1.x.array

    c1.name = 'conc'

    [fX, fY] = integrateFuidStress(uf, pf, mu, nVec, mesh, Gamma_CYL)
    cD = fX / (0.5 * rho * Ubar**2 * chord)
    cL = fY / (0.5 * rho * Ubar**2 * chord)

    time_Arr.append(t)
    fX_Arr.append(fX)
    fY_Arr.append(fY)
    cD_Arr.append(cD)
    cL_Arr.append(cL)

    el_c = integrateEluteFlux(c1, uf, D, nVec, mesh, Gamma_OUT)
    elute_Arr.append(el_c)

    print(f"t = {t:.6f}; Post-processing completed!")

    vFile.write_function(uf, t)
    pFile.write_function(pf, t)
    cFile.write_function(c1, t)

    print(f"t = {t:.6f}; Output files updated!")

    t += dt
    tn += 1

vFile.close()
pFile.close()
cFile.close()

#-------------------------------------------------------------------------------
# we will save the resulting time-history of forces and the fluxes as csv files
#-------------------------------------------------------------------------------
forceData = np.column_stack([time_Arr, fX_Arr, fY_Arr, cD_Arr, cL_Arr])
np.savetxt(forceFile, forceData, delimiter=',')

fluxData = np.column_stack([time_Arr, elute_Arr])
np.savetxt(fluxFile, fluxData, delimiter=',')

endTime = time.time()

print('Total simulation time:', endTime - startTime)

fig, ax = plt.subplots(1,2)
ax[0].plot(time_Arr[-200:], cD_Arr[-200:], 'r')
ax[1].plot(time_Arr[-200:], cL_Arr[-200:], 'm')
ax[0].set_xlabel('time', fontweight='bold')
ax[1].set_xlabel('time', fontweight='bold')
ax[0].set_ylabel('streamwise force', fontweight='bold')
ax[1].set_ylabel('cross-stream force', fontweight='bold')
plt.savefig(forceImage, bbox_inches='tight', dpi=120)
plt.close()

plt.plot(time_Arr, elute_Arr)
plt.xlabel('time', fontweight='bold')
plt.ylabel('eluted flux', fontweight='bold')
plt.savefig(eluteImage, bbox_inches='tight', dpi=120)
plt.close()