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


def checkLESMeshResolution(a_Mesh, a_Viscosity, a_Density, a_Ubar, a_Diam, histFile='mesh-les-check.png'):

    #
    # Step 1: project CellDiameter() → DG0 (one value per cell)
    #
    DG0 = dfx.fem.functionspace(a_Mesh, ("DG", 0))
    h_expr = dfx.fem.Expression(ufl.CellDiameter(a_Mesh), DG0.element.interpolation_points)
    h_func = dfx.fem.Function(DG0)
    h_func.interpolate(h_expr)

    #
    # Gather all ranks' cell data on rank 0
    #
    h_arr = a_Mesh.comm.gather(h_func.x.array.copy(), root=0)

    if a_Mesh.comm.rank == 0:

        h_arr = np.concatenate(h_arr)

        #
        # Step 2: LES length scales (a-priori, from problem parameters)
        #
        nu = a_Viscosity / a_Density    # kinematic viscosity   [m²/s]
        L_int = a_Diam                  # integral length scale  [m]
        eps_est = a_Ubar**3 / L_int       # dissipation estimate   [m²/s³]
        eta = (nu**3 / eps_est) ** 0.25 # Kolmogorov scale       [m]

        #
        # Step 3: per-cell ratios
        #
        ratio_eta = h_arr / eta
        ratio_L = h_arr / L_int

        #
        # Step 4: statistics
        #
        h_min = h_arr.min()
        h_max = h_arr.max()
        h_med = np.median(h_arr)

        re_min = ratio_eta.min()
        re_max = ratio_eta.max()
        re_med = np.median(ratio_eta)

        rl_min = ratio_L.min()
        rl_max = ratio_L.max()
        rl_med = np.median(ratio_L)

        THRESH_COARSE = 50.0
        THRESH_FINE = 10.0
        frac_coarse = np.mean(ratio_eta > THRESH_COARSE) * 100.0
        frac_fine = np.mean(ratio_eta < THRESH_FINE) * 100.0
        mesh_ok = (re_med < THRESH_COARSE) and (rl_med < 0.5)

        #
        # Step 5: printed report
        #
        sep = "=" * 62
        print(f"\n{sep}")
        print("  LES MESH RESOLUTION CHECK")
        print(sep)
        print(f"  Problem parameters")
        print(f"    Reference velocity   Ubar  = {a_Ubar:.4g}  m/s")
        print(f"    Reference length     L   = {L_int:.4g}  m  (cylinder chord)")
        print(f"    Kinematic viscosity  nu  = {nu:.4g}  m²/s")
        print(f"    Est. dissipation     eps = {eps_est:.4g}  m²/s³")
        print(f"    Kolmogorov scale     eta = {eta:.4g}  m")
        print(sep)
        print(f"  Cell chordeter  h  [m]")
        print(f"    min = {h_min:.4g}   median = {h_med:.4g}   max = {h_max:.4g}")
        print(sep)
        print(f"  Resolution ratio  h/eta  (target: 1 – {THRESH_COARSE:.0f})")
        print(f"    min = {re_min:.2f}   median = {re_med:.2f}   max = {re_max:.2f}")
        print(f"    Cells with h/eta > {THRESH_COARSE:.0f} (too coarse) : {frac_coarse:.1f}%")
        print(f"    Cells with h/eta < {THRESH_FINE:.0f}  (DNS-quality) : {frac_fine:.1f}%")
        print(sep)
        print(f"  Scale separation  h/L  (target: << 1)")
        print(f"    min = {rl_min:.4f}   median = {rl_med:.4f}   max = {rl_max:.4f}")
        print(sep)
        if mesh_ok:
            print("  VERDICT: ✓  Mesh appears ADEQUATE for LES with Smagorinsky.")
            print("              Median filter width sits in the inertial subrange.")
        else:
            print("  VERDICT: ✗  Mesh may be TOO COARSE for reliable LES.")
            if re_med >= THRESH_COARSE:
                print(f"              Median h/eta = {re_med:.1f} exceeds limit {THRESH_COARSE:.0f}.")
                print("              Consider refining the mesh or reducing Re.")
            if rl_med >= 0.5:
                print(f"              Median h/L = {rl_med:.3f} — large eddies under-resolved.")
        print(sep + "\n")

        #
        # Step 6: histogram
        #
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].hist(h_arr, bins=60, color='steelblue', edgecolor='white', linewidth=0.4)
        axes[0].axvline(h_med, color='red', linestyle='--', linewidth=1.5, label=f'median = {h_med:.4g} m')
        axes[0].axvline(eta,   color='orange', linestyle=':', linewidth=1.5, label=f'η = {eta:.4g} m')
        axes[0].set_xlabel('Cell chordeter (h) [m]', fontweight='bold')
        axes[0].set_ylabel('Number of cells', fontweight='bold')
        axes[0].set_title('Cell size distribution', fontweight='bold')
        axes[0].legend(fontsize=8)

        axes[1].hist(ratio_eta, bins=60, color='darkorange', edgecolor='white', linewidth=0.4)
        axes[1].axvline(re_med, color='red', linestyle='--', linewidth=1.5, label=f'median = {re_med:.1f}')
        axes[1].axvline(THRESH_COARSE, color='black', linestyle='-', linewidth=1.2, label=f'coarse limit = {THRESH_COARSE:.0f}')
        axes[1].axvline(THRESH_FINE, color='green', linestyle=':', linewidth=1.2, label=f'DNS limit = {THRESH_FINE:.0f}')
        axes[1].set_xlabel('h / η (resolution ratio)', fontweight='bold')
        axes[1].set_ylabel('Number of cells', fontweight='bold')
        axes[1].set_title('LES resolution ratio h/η', fontweight='bold')
        axes[1].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(histFile, bbox_inches='tight', dpi=120)
        plt.close()
        print(f"  Mesh resolution histogram saved to: {histFile}\n")

    else:
         mesh_ok = None

    # Broadcast the verdict so every rank gets a real bool
    mesh_ok = a_Mesh.comm.bcast(mesh_ok, root=0)
    return mesh_ok

def integrateFuidStress(a_U, a_P, a_MuEff, a_N, a_Mesh, a_GammaP):

    eps = 0.5*(ufl.grad(a_U) + ufl.grad(a_U).T)
    sig = -a_P*ufl.Identity(2) + 2.0*a_MuEff*eps

    traction = ufl.dot(sig, a_N)

    forceX = traction[0] * a_GammaP
    forceY = traction[1] * a_GammaP

    fXVal = dfx.fem.assemble_scalar(dfx.fem.form(forceX))
    fYVal = dfx.fem.assemble_scalar(dfx.fem.form(forceY))

    return [fXVal, fYVal]

def strainRate(vel):
    S = 0.5 * (ufl.grad(vel) + ufl.grad(vel).T)
    return S

def strainRateMag(vel):
    S = 0.5 * (ufl.grad(vel) + ufl.grad(vel).T)
    D = ufl.sqrt(2.0 * ufl.inner(S, S) + 1.0e-14)
    return D

# meshFile = 'geom-vonkarman-circle.msh'        # for laminar
outFileV = 'circle-V-LES.xdmf'
outFileP = 'circle-P-LES.xdmf'
forceFile = 'circle-forces-LES.dat'
imageFile = 'circle-draglift-LES.png'

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

            //Field[3].XMin = -0.1;
            //Field[3].XMax =  0.1;

            Field[3].XMin = {0.25*Chord - 0.1}; //Current changes
            Field[3].XMax = {0.25*Chord + 0.1}; //

            //Field[3].YMin = -0.1;
            //Field[3].YMax =  0.1;

            Field[3].YMin = {Height - 0.1};  //Current changes
            Field[3].YMax = {Height + 0.1};  //

            Field[4] = Box;
            Field[4].VIn  = E2;
            Field[4].VOut = E3;

            //Field[4].XMin = 0.9;
            //Field[4].XMax = 1.1;

            Field[4].XMin = {0.25*Chord + 0.9*Chord - 0.1}; //Current changes
            Field[4].XMax = {0.25*Chord + 0.9*Chord + 0.1}; //

            //Field[4].YMin = -0.1;
            //Field[4].YMax = 0.1;

            Field[4].YMin = {Height - 0.1}; //Current changes
            Field[4].YMax = {Height + 0.1}; //

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


viscosity = 0.002
density = 1.0
kinematic = viscosity/density
# Ubar = 2.5  # for laminar
Ubar = 15.0   # for LES
chord = 1
elemType = 'q1p1'
isSUPG = True
isPSPG = True
isLSIC = True


height = 1.5 #how high from ground (will likely get ranged)
AOA = 5 # degrees, also will get ranged
chord = 1 #scales chord of airfoil, leave at one for now
boxH = 8 * chord #sets height of bounding box for sim
boxL = 15 * chord #sets length of bounding box for sim


Inputs = [chord, height, AOA, boxH, boxL]
xyzFileName = sys.argv[1]
meshFile = writeGeoFromXYZVarySizing(xyzFileName, Inputs)


useLES = True
Cs = 0.17 # Smagorisnky constant (0.1 channel / 0.17 free shear)

# dt = 0.005    # for laminar
dt = 0.0005     # for LES
t_start = 0.0
# t_end = 7.0   # for laminar
t_end = 8.0     # for LES 
t_theta = 0.5

ID_AIRFOIL = 1
ID_INLET   = 2
ID_OUTLET  = 3
ID_TOP     = 4
ID_BOTTOM  = 5

Reynolds = (density * Ubar * chord) / kinematic
print("The Problem Reynolds Number is:", Reynolds)
print("LES model active:", useLES, " | Cs =", Cs if useLES else "N/A")

def noSlipBC(x):
    return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))

def pressureBC(x):
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

startTime = time.time()

mesh_data = gmshio.read_from_msh(meshFile, comm=MPI.COMM_WORLD, gdim=2)
mesh = mesh_data.mesh
cell_markers = mesh_data.cell_tags
facet_markers = mesh_data.facet_tags

nVec = ufl.FacetNormal(mesh)

tdim = mesh.topology.dim
fdim = tdim - 1

if useLES:
    mesh_ok = checkLESMeshResolution(mesh, viscosity, density, Ubar, chord, histFile='mesh-check.png')
    if not mesh_ok:
        print("  WARNING: Proceeding with a potentially under-resolved mesh.")
        print("           Results should be interpreted with caution.\n")

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
b_dofs_CYL = dfx.fem.locate_dofs_topological((W.sub(0),U_sub), fdim, facet_markers.find(ID_AIRFOIL))

uD_Wall = dfx.fem.Function(U_sub)
uD_Wall.interpolate(noSlipBC)

uD_Inlet = dfx.fem.Function(U_sub)
uD_Inlet.interpolate(inletBC)

uD_Outlet = dfx.fem.Function(P_sub)
uD_Outlet.interpolate(pressureBC)

uD_MovingWall = dfx.fem.Function(U_sub)
uD_MovingWall.interpolate(movingWallBC)

bc_INLET = dfx.fem.dirichletbc(uD_Inlet, b_dofs_INLET, W.sub(0))
bc_TOP = dfx.fem.dirichletbc(uD_Wall, b_dofs_TOP, W.sub(0))
bc_OUTLET = dfx.fem.dirichletbc(uD_Outlet, b_dofs_OUTLET, W.sub(1))
bc_BOTTOM = dfx.fem.dirichletbc(uD_MovingWall, b_dofs_BOTTOM, W.sub(0))
bc_CYL = dfx.fem.dirichletbc(uD_Wall, b_dofs_CYL, W.sub(0))

bc = [bc_INLET, bc_TOP, bc_OUTLET, bc_BOTTOM, bc_CYL]

ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_markers)
Gamma_CYL = ds(ID_AIRFOIL)

(v,q) = ufl.TestFunctions(W)

mu = dfx.fem.Constant(mesh, dfx.default_scalar_type(viscosity))
rho = dfx.fem.Constant(mesh, dfx.default_scalar_type(density))
idt = dfx.fem.Constant(mesh, dfx.default_scalar_type(1.0/dt))
theta = dfx.fem.Constant(mesh, dfx.default_scalar_type(t_theta))
b = dfx.fem.Constant(mesh, PETSc.ScalarType((0.0,0.0)))
I = ufl.Identity(2)

h = ufl.CellDiameter(mesh)

W1 = dfx.fem.Function(W)
(u,p) = ufl.split(W1)

if useLES:
    Cs_const = dfx.fem.Constant(mesh, dfx.default_scalar_type(Cs))
    mu_t1 = rho * (Cs_const * h) ** 2 * strainRateMag(u)
    mu_eff1 = mu + mu_t1
else:
    mu_eff1 = mu

strain_1 = strainRate(u) #0.5*(ufl.grad(u) + ufl.grad(u).T)
viscstress_1 = 2.0 * mu_eff1 * strain_1
stress_1 = -p*I + viscstress_1

T1_1 = rho * ufl.inner(v, ufl.grad(u)*u) * ufl.dx
T2_1 = ufl.inner(ufl.grad(v), stress_1) * ufl.dx
T3_1 = q * ufl.div(u) * ufl.dx
T4_1 = rho * ufl.dot(v,b) * ufl.dx
L_1  = T1_1 + T2_1 - T3_1 - T4_1

W0 = dfx.fem.Function(W)
(u0,p0) = ufl.split(W0)

if useLES:
    Cs_const = dfx.fem.Constant(mesh, dfx.default_scalar_type(Cs))
    mu_t0 = rho * (Cs_const * h) ** 2 * strainRateMag(u0)
    mu_eff0 = mu + mu_t0
else:
    mu_eff0 = mu

strain_0 = strainRate(u0) #0.5*(ufl.grad(u0) + ufl.grad(u0).T)
viscstress_0 = 2.0 * mu_eff0 * strain_0
stress_0 = -p*I + viscstress_0

T1_0 = rho * ufl.inner(v, ufl.grad(u0)*u0) * ufl.dx
T2_0 = ufl.inner(ufl.grad(v), stress_0) * ufl.dx
T3_0 = q * ufl.div(u0) * ufl.dx
T4_0 = rho * ufl.dot(v,b) * ufl.dx
L_0 = T1_0 + T2_0 - T3_0 - T4_0

F = idt * ufl.inner((u-u0),v) * ufl.dx + (1.0-theta) * L_0 + theta * L_1

uNorm = ufl.sqrt(ufl.inner(u0, u0))
tau = ( (2.0*theta*idt)**2 + (2.0*uNorm/h)**2 + (4.0*mu_eff0/h**2)**2 )**(-0.5)

residual = idt*rho*(u - u0) + \
    theta*(rho*ufl.grad(u)*u - ufl.div(stress_1) - rho*b) +\
    (1.0-theta)*(rho*ufl.grad(u0)*u0 - ufl.div(stress_0) - rho*b)

if isSUPG == True:
    F_SUPG = tau * ufl.inner(ufl.grad(v)*u, residual) * ufl.dx
    F = F + F_SUPG

if isPSPG == True:
    F_PSPG = - tau * ufl.inner(ufl.grad(q), residual) * ufl.dx
    F = F + F_PSPG

if isLSIC == True:
    F_LSIC = tau * ufl.inner(ufl.div(v), ufl.div(u)) * ufl.dx
    F = F + F_LSIC

problem = NewtonSolverNonlinearProblem(F, W1, bcs=bc)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-7
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

vFile = dfx.io.XDMFFile(mesh.comm, outFileV, "w", encoding=dfx.io.XDMFFile.Encoding.ASCII)
vFile.write_mesh(mesh)

pFile = dfx.io.XDMFFile(mesh.comm, outFileP, "w", encoding=dfx.io.XDMFFile.Encoding.ASCII)
pFile.write_mesh(mesh)

time_Arr = []
fX_Arr = []
fY_Arr = []
cD_Arr = []
cL_Arr = []

while t < t_end:

    t_in = time.time()
    n, converged = solver.solve(W1)
    assert (converged)
    t_out = time.time()

    print(f"t = {t:.6f}; Number of iterations: {n:d}; compute time: {t_out-t_in:f}")

    uf = W1.split()[0].collapse()
    pf = W1.split()[1].collapse()

    uf.name = 'vel'
    pf.name = 'pres'

    vFile.write_function(uf, t)
    pFile.write_function(pf, t)

    [fX, fY] = integrateFuidStress(uf, pf, mu_eff1, nVec, mesh, Gamma_CYL)
    cD = fX / (0.5 * rho * Ubar**2 * chord)
    cL = fY / (0.5 * rho * Ubar**2 * chord)

    time_Arr.append(t)
    fX_Arr.append(fX)
    fY_Arr.append(fY)
    cD_Arr.append(cD)
    cL_Arr.append(cL)

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
