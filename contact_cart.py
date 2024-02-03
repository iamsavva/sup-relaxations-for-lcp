import typing as T  # pylint: disable=unused-import

import numpy as np
import numpy.typing as npt

from pydrake.solvers import (  # pylint: disable=import-error, no-name-in-module, unused-import
    MakeSemidefiniteRelaxation,
    MathematicalProgram,
    MathematicalProgramResult,
    Solve,
    IpoptSolver,
    SnoptSolver,
    GurobiSolver,
    MosekSolver,
    MosekSolverDetails,
    SolverOptions,
    CommonSolverOption,
)

from pydrake.geometry.optimization import (  # pylint: disable=import-error, no-name-in-module
    HPolyhedron,
    Point,
    ConvexSet,
    GraphOfConvexSets,
    GraphOfConvexSetsOptions,
)

from pydrake.symbolic import ( # pylint: disable=import-error, no-name-in-module, unused-import
    Polynomial,
    Variable,
    Variables,
    Expression,
)  

from util import (
    timeit,
    diditwork,
    INFO,
    YAY,
    WARN,
    ERROR,
)  # pylint: disable=import-error, no-name-in-module, unused-import


class ContactCart:
    def __init__(self, N=10, h=0.1, m=1, x0=1, v0=-1, xt=2, vt=0, Qx=1, Qv=10, Ru=1, Rf=0) -> None:
        assert x0 >= 0
        self.N = N
        self.h = h
        self.m = m
        self.x0 = x0
        self.v0 = v0
        self.xt = xt
        self.vt = vt
        self.Qx = Qx
        self.Qv = Qv
        self.Ru = Ru
        self.Rf = Rf

    def form_nonlinear_prog(self) -> T.Tuple[MathematicalProgram, npt.NDArray,npt.NDArray,npt.NDArray,npt.NDArray ]:
        prog = MathematicalProgram()
        x = np.hstack( ([self.x0], prog.NewContinuousVariables(self.N, "x")) )
        v = np.hstack( ([self.v0], prog.NewContinuousVariables(self.N, "v")) )
        f = prog.NewContinuousVariables(self.N, "f")
        u = prog.NewContinuousVariables(self.N, "u")

        # cost at 0 point for completeness
        total_cost = Expression(0)
        total_cost += self.Qx * (x[0]-self.xt)**2
        total_cost += self.Qv * (v[0]-self.vt)**2
        prog.AddQuadraticCost( total_cost )

        for n in range(self.N):
            # constraints
            prog.AddLinearConstraint(x[n+1] >= 0)
            prog.AddLinearConstraint(f[n] >= 0)
            prog.AddLinearConstraint(x[n+1] == x[n] + self.h * v[n+1])
            prog.AddLinearConstraint(v[n+1] == v[n] + self.h/self.m * (u[n]+f[n]) )
            prog.AddConstraint(x[n+1]*f[n] == 0)

            # quadratic cost on each term
            total_cost = Expression(0)
            total_cost += self.Qx * (x[n+1]-self.xt)**2
            total_cost += self.Qv * (v[n+1]-self.vt)**2
            total_cost += self.Ru * (u[n]-0)**2
            total_cost += self.Rf * (f[n]-0)**2
            prog.AddQuadraticCost( total_cost )
        
        return prog,x,v,f,u
    
    def solve_with_ipopt(self):
        prog,x,v,f,u = self.form_nonlinear_prog()

        timer = timeit()
        solver = IpoptSolver()
        solution = solver.Solve(prog) # type: MathematicalProgramResult
        timer.dt("IPOPT")
        diditwork(solution)
        INFO("-------------------")

    def solve_with_snopt(self):
        prog,x,v,f,u = self.form_nonlinear_prog()
        timer = timeit()
        solver = SnoptSolver()
        solution = solver.Solve(prog) # type: MathematicalProgramResult
        timer.dt("SNOPT")
        diditwork(solution)
        INFO("-------------------")

    def solve_nonconvex_qcqp_sdp_relaxation(self):
        prog,x,v,f,u = self.form_nonlinear_prog()
        sdp_prog = MakeSemidefiniteRelaxation(prog)
        timer = timeit()
        solver = MosekSolver()
        solution = solver.Solve(sdp_prog) # type: MathematicalProgramResult
        timer.dt("nonconvex QCQP relaxaion with MOSEK")
        diditwork(solution)
        INFO("-------------------")

    def build_gcs(self)-> T.Tuple[GraphOfConvexSets, GraphOfConvexSets.Vertex, GraphOfConvexSets.Vertex]:
        gcs = GraphOfConvexSets()
        vertices = dict() # type: T.Dict[str, GraphOfConvexSets.Vertex]
        umax = 100
        fmax = 100
        xmax = 100
        vmax = 100
        if self.x0 > 0:
            fmax_0 = 0
        else:
            fmax_0 = fmax

        cost_mat = np.diag([self.Qx, self.Qv, self.Ru, self.Rf])
        final_cost_mat = np.diag([self.Qx, self.Qv])
        x_star = np.array([self.xt, self.vt, 0, 0])
        final_x_star = np.array([self.xt, self.vt])
        
        v0 = gcs.AddVertex(HPolyhedron.MakeBox([self.x0, self.v0, -umax, 0],[self.x0, self.v0, umax, fmax_0]), "v_0")
        v0.AddCost( (v0.x()-x_star).dot(cost_mat).dot(v0.x()-x_star) )

        vN = gcs.AddVertex(HPolyhedron.MakeBox([0, -vmax],[xmax, vmax]), "v_"+str(self.N))
        vN.AddCost( (vN.x()-final_x_star).dot(final_cost_mat).dot(vN.x()-final_x_star) )

        for n in range(1, self.N):
            # in contact, position is 0, force is positive
            vc_n = gcs.AddVertex(HPolyhedron.MakeBox([0, -vmax, -umax, 0],[0, vmax, umax, fmax]), "vc_"+str(n))
            vc_n.AddCost( (vc_n.x()-x_star).dot(cost_mat).dot(vc_n.x()-x_star) )
            vertices["vc_"+str(n)] = vc_n
            
            # no contact, position is positive, force is 0
            vnc_n = gcs.AddVertex(HPolyhedron.MakeBox([0, -vmax, -umax, 0],[xmax, vmax, umax, 0]), "vnc_"+str(n))
            vnc_n.AddCost( (vnc_n.x()-x_star).dot(cost_mat).dot(vnc_n.x()-x_star) )
            vertices["vnc_"+str(n)] = vnc_n

        
        def add_cons(e:GraphOfConvexSets.Edge):
            h,m = self.h, self.m
            e.AddConstraint(e.xv()[0] == e.xu()[0] + h*e.xv()[1] )
            e.AddConstraint(e.xv()[1] == e.xu()[1] + h / m * ( e.xu()[2] + e.xu()[3]) )

        
        ec = gcs.AddEdge(v0, vertices["vc_1"], "0->c_1")
        add_cons(ec)
        en = gcs.AddEdge(v0, vertices["vnc_1"], "0->nc_1")
        add_cons(en)

        for k in range(1, self.N-1):
            n = str(k)
            n1 = str(k+1)
            ecc = gcs.AddEdge(vertices["vc_"+n], vertices["vc_"+n1], "c_"+n+"->c_"+n1)
            ecn = gcs.AddEdge(vertices["vc_"+n], vertices["vnc_"+n1], "c_"+n+"->nc_"+n1)
            enn = gcs.AddEdge(vertices["vnc_"+n], vertices["vnc_"+n1], "nc_"+n+"->nc_"+n1)
            enc = gcs.AddEdge(vertices["vnc_"+n], vertices["vc_"+n1], "nc_"+n+"->c_"+n1)
            add_cons(ecc)
            add_cons(ecn)
            add_cons(enn)
            add_cons(enc)

        n = str(self.N-1)
        ec = gcs.AddEdge(vertices["vc_"+n], vN, "c_"+n+"->"+str(self.N) )
        add_cons(ec)
        en = gcs.AddEdge(vertices["vnc_"+n], vN, "nc_"+n+"->"+str(self.N))
        add_cons(en)
        return gcs, v0, vN


    def solve_cr_with_gcs(self):
        gcs, v0, vN = self.build_gcs()
        options = GraphOfConvexSetsOptions()
        options.convex_relaxation = True
        options.max_rounded_paths = 0
        timer = timeit()
        solution = gcs.SolveShortestPath(v0, vN, options)
        timer.dt("GCS CR")
        diditwork(solution)
        INFO("-------------------")



    def solve_micp_with_gcs(self):
        gcs, v0, vN = self.build_gcs()
        options = GraphOfConvexSetsOptions()
        options.convex_relaxation = False
        timer = timeit()
        solution = gcs.SolveShortestPath(v0, vN, options)
        timer.dt("GCS MICP")
        diditwork(solution)
        INFO("-------------------")

    def solve_sdp_convex_diff_decomposition(self):
        def add_sdp_constraints(prog:MathematicalProgram, x:npt.NDArray, y:npt.NDArray):
            # add SDP relaxation of constraint x^T y = 0
            n = len(x)
            p = x+y
            q = x-y
            P = prog.NewSymmetricContinuousVariables(n, "P")
            Q = prog.NewSymmetricContinuousVariables(n, "Q")
            prog.AddLinearConstraint( np.trace(P) - np.trace(Q) == 0)
            P_mat = np.vstack((np.hstack((P, p.reshape((n,1)))), np.hstack((p, [1]))))
            Q_mat = np.vstack((np.hstack((Q, q.reshape((n,1)))), np.hstack((q, [1]))))
            prog.AddPositiveSemidefiniteConstraint( P_mat )
            prog.AddPositiveSemidefiniteConstraint( Q_mat )

            # add quadratic cost on P

            
            






cart = ContactCart()
cart.solve_with_ipopt()
cart.solve_nonconvex_qcqp_sdp_relaxation()
cart.solve_cr_with_gcs()
cart.solve_micp_with_gcs()