import basix
from mpi4py import MPI
import dolfinx as dfx
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

def integrateFuidStress(a_U, a_P, a_Mu, a_N, a_Mesh, a_GammaP):

    eps = 0.5*(ufl.grad(a_U) + ufl.grad(a_U).T)
    sig = -a_P*ufl.Identity(2) + 2.0*a_Mu*eps

    traction = ufl.dot(sig, a_N)

    forceX = traction[0] * a_GammaP
    forceY = traction[1] * a_GammaP

    fXVal = dfx.fem.assemble_scalar(dfx.fem.form(forceX))
    fYVal = dfx.fem.assemble_scalar(dfx.fem.form(forceY))

    return [fXVal, fYVal]

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

viscosity = 1.48e-5 #dynamic viscosity, (m^2)/s
density = 1.225 #kg/m^3
U0_target = 15.0   # final desired freestream velocity
U0 = 2.0           # start with a small, stable value
U0_ramp_rate = 0.2 # how fast we increase it (m/s per time unit)
height = 2 #how high from ground (will likely get ranged)
AOA = 5 # degrees, also will get ranged
chord = 1 #scales chord of airfoil, leave at one for now
boxH = 4.1 * chord #sets height of bounding box for sim
boxL = 22.0 * chord #sets length of bounding box for sim
elemType = 'q1p1'
isStabilize = True

Inputs = [chord, height, AOA, boxH, boxL]
xyzFileName = sys.argv[1]
meshfile = writeGeoFromXYZVarySizing(xyzFileName, Inputs)


# number of particles
N = 10

# create a line of seeds in front of airfoil
x = np.linspace(-0.5, -0.2, N)
y = np.zeros(N)
z = np.zeros(N)

seeds = np.column_stack([x, y, z])

np.savetxt("seeds.csv", seeds, delimiter=",")

particleFile = 'seeds.csv'
outFileV = 'circle-V.xdmf'
outFileP = 'circle-P.xdmf'
forceFile = 'circle-forces.dat'
imageFile = 'circle-draglift.png'



dt = 0.0002
t_start = 0.0
t_end = 5.0
t_theta = 0.5
injectEvery = 50

ID_AIRFOIL = 1
ID_INLET   = 2
ID_OUTLET  = 3
ID_TOP     = 4
ID_BOTTOM  = 5

Reynolds = (U0 * chord) / viscosity
print("The Problem Reynolds Number is:", Reynolds)

def noSlipBC(x):
    return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))

def pressureBC(x):
    return np.zeros(x.shape[1])

def inletBC(x):
    vals = np.zeros((mesh.geometry.dim, x.shape[1]))
    vals[0] = U0 #constant vel
    vals[1] = 0.0
    return vals

def movingWallBC(x):
    vals = np.zeros((mesh.geometry.dim, x.shape[1]))
    vals[0] = U0   # moving with freestream
    vals[1] = 0.0
    return vals

def slipWallBC(x):
    vals = np.zeros((mesh.geometry.dim, x.shape[1]))
    vals[1] = 0.0   # no penetration only
    return vals

startTime = time.time()

result = gmshio.read_from_msh(meshfile, MPI.COMM_WORLD, gdim=2)

mesh = result[0]
cell_markers = result[1]
facet_markers = result[2]

nVec = ufl.FacetNormal(mesh)
tdim = mesh.topology.dim
fdim = tdim - 1

particles = np.genfromtxt(particleFile, delimiter=',', skip_header=1, dtype=np.float64)
numParticles = particles.shape[0]
print("number of particles", numParticles)
seedParticles = particles.copy()

PE = basix.ufl.element('Lagrange', mesh.basix_cell(), degree=1)

if elemType == 'q2p1':
    QE = basix.ufl.element('Lagrange', mesh.basix_cell(), degree=2, shape=(mesh.topology.dim,))
elif elemType == 'q1p1':
    QE = basix.ufl.element('Lagrange', mesh.basix_cell(), degree=1, shape=(mesh.topology.dim,))

ME = basix.ufl.mixed_element([QE, PE])
W = dfx.fem.functionspace(mesh, ME)

U_sub, U_submap = W.sub(0).collapse()
P_sub, P_submap = W.sub(1).collapse()

b_dofs_INLET = dfx.fem.locate_dofs_topological((W.sub(0),U_sub), fdim, facet_markers.find(ID_INLET))
b_dofs_TOP = dfx.fem.locate_dofs_topological((W.sub(0),U_sub), fdim, facet_markers.find(ID_TOP))
b_dofs_OUTLET = dfx.fem.locate_dofs_topological((W.sub(1),P_sub), fdim, facet_markers.find(ID_OUTLET))
b_dofs_BOTTOM = dfx.fem.locate_dofs_topological((W.sub(0),U_sub), fdim, facet_markers.find(ID_BOTTOM))
b_dofs_AIRFOIL = dfx.fem.locate_dofs_topological((W.sub(0),U_sub), fdim, facet_markers.find(ID_AIRFOIL))

uD_Wall = dfx.fem.Function(U_sub)
uD_Wall.interpolate(noSlipBC)

uD_MovingWall = dfx.fem.Function(U_sub)
uD_MovingWall.interpolate(movingWallBC)

uD_Inlet = dfx.fem.Function(U_sub)
uD_Inlet.interpolate(inletBC)

uD_Outlet = dfx.fem.Function(P_sub)
uD_Outlet.interpolate(pressureBC)

bc_INLET = dfx.fem.dirichletbc(uD_Inlet, b_dofs_INLET, W.sub(0))
bc_OUTLET = dfx.fem.dirichletbc(uD_Outlet, b_dofs_OUTLET, W.sub(1))
bc_BOTTOM = dfx.fem.dirichletbc(uD_MovingWall, b_dofs_BOTTOM, W.sub(0))
bc_AIRFOIL = dfx.fem.dirichletbc(uD_Wall, b_dofs_AIRFOIL, W.sub(0))

bc = [bc_INLET, bc_OUTLET, bc_BOTTOM, bc_AIRFOIL]

ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_markers)
Gamma_AIRFOIL = ds(ID_AIRFOIL)

(v,q) = ufl.TestFunctions(W)

mu = dfx.fem.Constant(mesh, dfx.default_scalar_type(viscosity))
rho = dfx.fem.Constant(mesh, dfx.default_scalar_type(density))
idt = dfx.fem.Constant(mesh, dfx.default_scalar_type(1.0/dt))
theta = dfx.fem.Constant(mesh, dfx.default_scalar_type(t_theta))
b = dfx.fem.Constant(mesh, PETSc.ScalarType((0.0,0.0)))
I = ufl.Identity(2)

W1 = dfx.fem.Function(W)
(u,p) = ufl.split(W1)

T1_1 = rho * ufl.inner(v, ufl.grad(u)*u) * ufl.dx
T2_1 = mu * ufl.inner(ufl.grad(v), ufl.grad(u)) * ufl.dx
T3_1 = p * ufl.div(v) * ufl.dx
T4_1 = q * ufl.div(u) * ufl.dx
T5_1 = rho * ufl.dot(v,b) * ufl.dx
L_1  = T1_1 + T2_1 - T3_1 -T4_1 - T5_1

W0 = dfx.fem.Function(W)
(u0,p0) = ufl.split(W0)

T1_0 = rho * ufl.inner(v, ufl.grad(u0)*u0) * ufl.dx
T2_0 = mu * ufl.inner(ufl.grad(v), ufl.grad(u0)) * ufl.dx
T3_0 = p * ufl.div(v) * ufl.dx
T4_0 = q * ufl.div(u0) * ufl.dx
T5_0 = rho * ufl.dot(v,b) * ufl.dx
L_0 = T1_0 + T2_0 - T3_0 -T4_0 - T5_0

F = idt * ufl.inner((u-u0),v) * ufl.dx + (1.0-theta) * L_0 + theta * L_1

#May need to adjust tuning, its for a cylinder not for an airfoil

uNorm = ufl.sqrt(ufl.inner(u0, u0))
h = ufl.CellDiameter(mesh)
tau = 1.0 / (idt + uNorm/h + 4.0*mu/(h*h))
u_mid = theta*u + (1.0-theta)*u0

residual = (
    idt*rho*(u - u0)
    + rho*ufl.grad(u_mid)*u_mid
    - mu*ufl.div(ufl.grad(u))
    + ufl.grad(p)
    - rho*b
)

F_SUPG = tau * ufl.inner(ufl.grad(v)*u_mid, residual) * ufl.dx

if isStabilize:
    F_PSPG = - tau * ufl.inner(ufl.grad(q), residual) * ufl.dx

F = F + F_SUPG + F_PSPG

#End of tuning section
problem = NewtonSolverNonlinearProblem(F, W1, bcs=bc)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-7
solver.damping = 0.3
solver.max_it = 20
solver.report = True

ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "gmres"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

t = t_start
tn = 0

if outFileV.endswith('.pvd'):
    vFile = dfx.io.VTKFile(MPI.COMM_WORLD, outFileV, "w")
elif outFileV.endswith('.xdmf'):
    vFile = dfx.io.XDMFFile(mesh.comm, outFileV, "w", encoding=dfx.io.XDMFFile.Encoding.ASCII)
    vFile.write_mesh(mesh)

if outFileP.endswith('.pvd'):
    pFile = dfx.io.VTKFile(MPI.COMM_WORLD, outFileP, "w")
elif outFileP.endswith('.xdmf'):
    pFile = dfx.io.XDMFFile(mesh.comm, outFileP, "w", encoding=dfx.io.XDMFFile.Encoding.ASCII)
    pFile.write_mesh(mesh)

time_Arr = []
fX_Arr = []
fY_Arr = []
cD_Arr = []
cL_Arr = []
pt = np.zeros(mesh.geometry.dim, dtype=np.float64)
tree = dfx.geometry.bb_tree(mesh, mesh.topology.dim)

while t < t_end:

    if U0 < U0_target:
        U0 += U0_ramp_rate * dt
        U0 = min(U0, U0_target)

        uD_Inlet.interpolate(inletBC)
        uD_MovingWall.interpolate(movingWallBC)



    t_in = time.time()
    n, converged = solver.solve(W1)
    assert (converged)
    t_out = time.time()

    print(f"t = {t:.6f}; Number of iterations: {n:d}; compute time: {t_out-t_in:f}")

    uf = W1.split()[0].collapse()
    pf = W1.split()[1].collapse()

    uf.name = 'vel'
    pf.name = 'pres'

    if outFileV.endswith('.pvd'):
        vFile.write_function(uf, tn)
    elif outFileV.endswith('.xdmf'):
        vFile.write_function(uf, tn)

    if outFileP.endswith('.pvd'):
        pFile.write_function(pf, tn)
    elif outFileP.endswith('.xdmf'):
        pFile.write_function(pf, tn)

    [fX, fY] = integrateFuidStress(uf, pf, mu, nVec, mesh, Gamma_AIRFOIL)
    cD = fX / (0.5 * rho * U0**2 * chord)
    cL = fY / (0.5 * rho * U0**2 * chord)

    time_Arr.append(t)
    fX_Arr.append(fX)
    fY_Arr.append(fY)
    cD_Arr.append(cD)
    cL_Arr.append(cL)


    if ((t > t_start) and ((tn % injectEvery) == 0)):
        particles = np.concatenate((particles, seedParticles), axis=0)
        numParticles = particles.shape[0]

    for i in range(numParticles):

        pt = particles[i,:]
        cells = dfx.geometry.compute_collisions_points(tree, pt)
        colliding_cells = dfx.geometry.compute_colliding_cells(mesh, cells, pt)
        cell_candidates = colliding_cells.links(0)

        if len(cell_candidates) > 0:
            cell = cell_candidates[0]
            vel_ploc = uf.eval(pt, cell)
            particles[i,0] = particles[i,0] + vel_ploc[0]*dt
            particles[i,1] = particles[i,1] + vel_ploc[1]*dt
        else:
            print(f"Point {pt} is not in domain.")

    lagrangianFile = particleFile.split('.')[0]+'-'+str(tn)+'.csv'
    np.savetxt(lagrangianFile, particles, delimiter=',', header="x, y, z")

    W0.x.array[:] = W1.x.array

    t += dt
    tn += 1

vFile.close()
pFile.close()

forceData = np.column_stack([time_Arr, fX_Arr, fY_Arr, cD_Arr, cL_Arr])
np.savetxt(forceFile, forceData)

endTime = time.time()

print('Total simulation time:', endTime - startTime)

fig, ax = plt.subplots(1,2)
ax[0].plot(time_Arr[-200:], cD_Arr[-200:], 'r')
ax[1].plot(time_Arr[-200:], cL_Arr[-200:], 'm')
ax[0].set_xlabel('time', fontweight='bold')
ax[1].set_xlabel('time', fontweight='bold')
ax[0].set_ylabel('streamwise force', fontweight='bold')
ax[1].set_ylabel('cross-stream force', fontweight='bold')
plt.savefig(imageFile, bbox_inches='tight',dpi=120)
plt.close()
