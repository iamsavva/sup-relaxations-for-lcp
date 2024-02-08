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
    Spectrahedron,
)

from pydrake.symbolic import (  # pylint: disable=import-error, no-name-in-module, unused-import
    Polynomial,
    Variable,
    Variables,
    Expression,
)

from pydrake.math import eq, le, ge

from util import (
    timeit,
    diditwork,
    INFO,
    YAY,
    WARN,
    ERROR,
)  # pylint: disable=import-error, no-name-in-module, unused-import


class ContactCart:
    def __init__(
        self,
        N=10,
        h=0.1,
        m=1,
        x0=1,
        v0=-1,
        xt=2,
        vt=0,
        Qx=1,
        Qv=0.01,
        Ru=0.01,
        Rf=0.0,
        umax=10,
        verbose=True,
        qcqp_scaling_factor=0.01,
    ) -> None:
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
        self.qcqp_scaling_factor = qcqp_scaling_factor
        self.verbose = verbose
        self.umax = umax

    def form_nonlinear_prog(
        self,
    ) -> T.Tuple[
        MathematicalProgram, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray
    ]:
        prog = MathematicalProgram()
        x = np.hstack(([self.x0], prog.NewContinuousVariables(self.N, "x")))
        v = np.hstack(([self.v0], prog.NewContinuousVariables(self.N, "v")))
        f = prog.NewContinuousVariables(self.N, "f")
        u = prog.NewContinuousVariables(self.N, "u")

        # cost at 0 point for completeness
        total_cost = Expression(0)
        total_cost += self.Qx * (x[0] - self.xt) ** 2
        total_cost += self.Qv * (v[0] - self.vt) ** 2
        prog.AddQuadraticCost(total_cost)

        for n in range(self.N):
            # constraints
            prog.AddLinearConstraint(x[n + 1] >= 0)
            prog.AddLinearConstraint(f[n] >= 0)
            prog.AddLinearConstraint(x[n + 1] == x[n] + self.h * v[n + 1])
            prog.AddLinearConstraint(v[n + 1] == v[n] + self.h / self.m * (u[n] + f[n]))
            prog.AddConstraint(x[n + 1] * f[n] == 0)
            prog.AddLinearConstraint(u[n] <= self.umax)

            # quadratic cost on each term
            total_cost = Expression(0)
            total_cost += self.Qx * (x[n + 1] - self.xt) ** 2
            total_cost += self.Qv * (v[n + 1] - self.vt) ** 2
            total_cost += self.Ru * (u[n] - 0) ** 2
            total_cost += self.Rf * (f[n] - 0) ** 2
            prog.AddQuadraticCost(total_cost)

        return prog, x, v, f, u

    def get_lcp_violation(self, x: npt.NDArray, f: npt.NDArray):
        return x[1:] * f

    def get_constraint_violation(
        self, x: npt.NDArray, v: npt.NDArray, f: npt.NDArray, u: npt.NDArray
    ):
        lcp = x[1:] * f
        pos = abs(x[1:] - x[:-1] - self.h * v[1:])
        vel = abs(v[1:] - v[:-1] - self.h / self.m * (u + f))
        return lcp + pos + vel

    def solve_with_ipopt(
        self,
    ) -> T.Tuple[
        float, float, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray
    ]:
        prog, x, v, f, u = self.form_nonlinear_prog()

        timer = timeit()
        solver = IpoptSolver()
        solution = solver.Solve(prog)  # type: MathematicalProgramResult
        solve_time = timer.dt("IPOPT", verbose=self.verbose)
        diditwork(solution, verbose=self.verbose)
        INFO("-------------------", verbose=self.verbose)
        cost = solution.get_optimal_cost()
        ev = np.vectorize(lambda a: a.Evaluate())
        x_traj = ev(solution.GetSolution(x).reshape(self.N + 1))
        v_traj = ev(solution.GetSolution(v).reshape(self.N + 1))
        f_traj = solution.GetSolution(f)
        u_traj = solution.GetSolution(u)
        lcp_v_traj = self.get_lcp_violation(x_traj, f_traj)
        return solve_time, cost, x_traj, v_traj, f_traj, u_traj, lcp_v_traj

    def warmstart_with_ipopt(
        self, x_w, v_w, f_w, u_w
    ) -> T.Tuple[
        float, float, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray
    ]:
        prog, x, v, f, u = self.form_nonlinear_prog()

        prog.SetInitialGuess(x[1:], x_w[1:])
        prog.SetInitialGuess(v[1:], v_w[1:])
        prog.SetInitialGuess(f, f_w)
        prog.SetInitialGuess(u, u_w)

        timer = timeit()
        solver = IpoptSolver()
        solution = solver.Solve(prog)  # type: MathematicalProgramResult
        solve_time = timer.dt("IPOPT with warmstart", verbose=self.verbose)
        diditwork(solution, verbose=self.verbose)
        INFO("-------------------", verbose=self.verbose)
        cost = solution.get_optimal_cost()
        ev = np.vectorize(lambda a: a.Evaluate())
        x_traj = ev(solution.GetSolution(x).reshape(self.N + 1))
        v_traj = ev(solution.GetSolution(v).reshape(self.N + 1))
        f_traj = solution.GetSolution(f)
        u_traj = solution.GetSolution(u)
        lcp_v_traj = self.get_lcp_violation(x_traj, f_traj)
        return solve_time, cost, x_traj, v_traj, f_traj, u_traj, lcp_v_traj

    def warmstart_with_snopt(
        self, x_w, v_w, f_w, u_w
    ) -> T.Tuple[
        float, float, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray
    ]:
        prog, x, v, f, u = self.form_nonlinear_prog()

        prog.SetInitialGuess(x[1:], x_w[1:])
        prog.SetInitialGuess(v[1:], v_w[1:])
        prog.SetInitialGuess(f, f_w)
        prog.SetInitialGuess(u, u_w)

        timer = timeit()
        solver = SnoptSolver()
        solution = solver.Solve(prog)  # type: MathematicalProgramResult
        solve_time = timer.dt("SNOPT with warmstart", verbose=self.verbose)
        diditwork(solution, verbose=self.verbose)
        INFO("-------------------", verbose=self.verbose)
        cost = solution.get_optimal_cost()
        ev = np.vectorize(lambda a: a.Evaluate())
        x_traj = ev(solution.GetSolution(x).reshape(self.N + 1))
        v_traj = ev(solution.GetSolution(v).reshape(self.N + 1))
        f_traj = solution.GetSolution(f)
        u_traj = solution.GetSolution(u)
        lcp_v_traj = self.get_lcp_violation(x_traj, f_traj)
        return solve_time, cost, x_traj, v_traj, f_traj, u_traj, lcp_v_traj

    def solve_with_snopt(
        self,
    ) -> T.Tuple[
        float, float, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray
    ]:
        prog, x, v, f, u = self.form_nonlinear_prog()
        timer = timeit()
        solver = SnoptSolver()
        solution = solver.Solve(prog)  # type: MathematicalProgramResult
        solve_time = timer.dt("SNOPT", verbose=self.verbose)
        diditwork(solution, verbose=self.verbose)
        INFO("-------------------", verbose=self.verbose)
        cost = solution.get_optimal_cost()
        ev = np.vectorize(lambda a: a.Evaluate())
        x_traj = ev(solution.GetSolution(x).reshape(self.N + 1))
        v_traj = ev(solution.GetSolution(v).reshape(self.N + 1))
        f_traj = solution.GetSolution(f)
        u_traj = solution.GetSolution(u)
        lcp_v_traj = self.get_lcp_violation(x_traj, f_traj)
        return solve_time, cost, x_traj, v_traj, f_traj, u_traj, lcp_v_traj

    def solve_nonconvex_qcqp_sdp_relaxation(
        self,
    ) -> T.Tuple[
        float, float, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray
    ]:
        # that's the usual SDP relaxation
        prog, x, v, f, u = self.form_nonlinear_prog()
        sdp_prog = MakeSemidefiniteRelaxation(prog)
        timer = timeit()
        solver = MosekSolver()
        solution = solver.Solve(sdp_prog)  # type: MathematicalProgramResult
        solve_time = timer.dt(
            "nonconvex QCQP relaxaion with MOSEK", verbose=self.verbose
        )
        diditwork(solution, verbose=self.verbose)
        INFO("-------------------", verbose=self.verbose)
        cost = solution.get_optimal_cost()
        ev = np.vectorize(lambda a: a.Evaluate())
        x_traj = ev(solution.GetSolution(x).reshape(self.N + 1))
        v_traj = ev(solution.GetSolution(v).reshape(self.N + 1))
        f_traj = solution.GetSolution(f)
        u_traj = solution.GetSolution(u)
        lcp_v_traj = self.get_constraint_violation(x_traj, v_traj, f_traj, u_traj)

        return solve_time, ost, x_traj, v_traj, f_traj, u_traj, lcp_v_traj

    def solve_chordal_nonconvex_qcqp_sdp_relaxation(
        self,
    ) -> T.Tuple[
        float, float, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray
    ]:
        # that's the usual SDP relaxation, but with chordal sparsity.
        # this requires some GCS trickery to produce
        gcs = GraphOfConvexSets()

        def extract_moments_from_spectrahedron_prog_variables(
            vector: npt.NDArray, dim: int
        ) -> T.Tuple[float, npt.NDArray, npt.NDArray]:
            # spectrahedron program variables stores vectors in the following array: m1, m2[triu], m0
            # indecis
            ind1 = dim
            ind2 = ind1 + int(dim * (dim + 1) / 2)
            # get moments
            m1 = vector[:ind1]
            m2 = np.zeros((dim, dim), dtype=vector.dtype)
            triu_ind = np.triu_indices(dim)
            m2[triu_ind[0], triu_ind[1]] = vector[ind1:ind2]
            m2[triu_ind[1], triu_ind[0]] = vector[ind1:ind2]
            m0 = vector[ind2]
            return m0, m1, m2

        prev_v = None
        # for each vertex
        for n in range(self.N):
            # generate a vertex for n -> n+1
            prog = MathematicalProgram()
            x1 = prog.NewContinuousVariables(1)[0]
            v1 = prog.NewContinuousVariables(1)[0]

            f1 = prog.NewContinuousVariables(1)[0]
            u1 = prog.NewContinuousVariables(1)[0]
            x2 = prog.NewContinuousVariables(1)[0]
            v2 = prog.NewContinuousVariables(1)[0]
            # add constraints on vertex
            if n == 0:
                prog.AddLinearConstraint(x1 == self.x0)
                prog.AddLinearConstraint(v1 == self.v0)
            prog.AddLinearConstraint(x2 >= 0)
            prog.AddLinearConstraint(f1 >= 0)
            prog.AddLinearConstraint(x2 == x1 + self.h * v2)
            prog.AddLinearConstraint(v2 == v1 + self.h / self.m * (u1 + f1))
            prog.AddConstraint(x2 * f1 == 0)
            prog.AddLinearConstraint(u1 <= self.umax)
            sdp_prog = MakeSemidefiniteRelaxation(prog)
            spectrahedron = Spectrahedron(sdp_prog)
            v = gcs.AddVertex(spectrahedron, str(n))
            # add edge constraints
            if n > 0:
                e = gcs.AddEdge(prev_v, v, str(n - 1) + "->" + str(n))
                _, vu1, vu2 = extract_moments_from_spectrahedron_prog_variables(
                    e.xu(), 6
                )
                _, vv1, vv2 = extract_moments_from_spectrahedron_prog_variables(
                    e.xv(), 6
                )
                cons = eq(vu1[4:6], vv1[0:2])
                for con in cons:
                    e.AddConstraint(con)
                cons = eq(vu2[4:6, 4:6], vv2[0:2, 0:2]).reshape(4)
                for con in cons:
                    e.AddConstraint(con)

            # add cost
            m0, m1, m2 = extract_moments_from_spectrahedron_prog_variables(v.x(), 6)
            cost = Expression(0)
            if n == 0:
                cost += (
                    m2[0, 0] * self.Qx
                    + m0 * self.Qx * self.xt**2
                    - 2 * self.Qx * self.xt * m1[0]
                )
                # print(m0 * self.Qx * self.xt **2)
                # print(2 * self.Qx * self.xt * m1[0])
                cost += (
                    m2[1, 1] * self.Qv
                    + m0 * self.Qv * self.vt**2
                    - 2 * self.Qv * self.vt * m1[1]
                )

            cost += m2[2, 2] * self.Rf
            cost += m2[3, 3] * self.Ru
            cost += (
                m2[4, 4] * self.Qx
                + m0 * self.Qx * self.xt**2
                - 2 * self.Qx * self.xt * m1[4]
            )
            cost += (
                m2[5, 5] * self.Qv
                + m0 * self.Qv * self.vt**2
                - 2 * self.Qv * self.vt * m1[5]
            )
            v.AddCost(cost)

            if n == 0:
                vs = v

            prev_v = v

        options = GraphOfConvexSetsOptions()
        options.convex_relaxation = True
        options.max_rounded_paths = 10
        timer = timeit()
        solution = gcs.SolveShortestPath(vs, v, options)
        solve_time = timer.dt(
            "chordal noconvex-qcqp SDP relaxation", verbose=self.verbose
        )
        diditwork(solution, verbose=self.verbose)
        INFO("-------------------", verbose=self.verbose)

        cost = solution.get_optimal_cost()
        edge_traj = gcs.GetSolutionPath(vs, v, solution)
        vertex_traj = []
        for edge in edge_traj:
            m0, m1, m2 = extract_moments_from_spectrahedron_prog_variables(
                solution.GetSolution(edge.u().x()), 6
            )
            vertex_traj.append(m1[0:4])
        m0, m1, m2 = extract_moments_from_spectrahedron_prog_variables(
            solution.GetSolution(edge_traj[-1].v().x()), 6
        )
        vertex_traj.append(m1[4:6])

        x_traj = np.array([x[0] for x in vertex_traj])
        v_traj = np.array([x[1] for x in vertex_traj])
        u_traj = np.array([x[3] for x in vertex_traj[:-1]])
        f_traj = np.array([x[2] for x in vertex_traj[:-1]])
        lcp_v_traj = self.get_lcp_violation(x_traj, f_traj)

        return solve_time, cost, x_traj, v_traj, f_traj, u_traj, lcp_v_traj

    def build_gcs(
        self,
    ) -> T.Tuple[GraphOfConvexSets, GraphOfConvexSets.Vertex, GraphOfConvexSets.Vertex]:
        # x v u f
        gcs = GraphOfConvexSets()
        vertices = dict()  # type: T.Dict[str, GraphOfConvexSets.Vertex]
        umax = self.umax
        fmax = 100
        xmax = 100
        vmax = 100
        if self.x0 > 0:
            fmax_0 = 100
        else:
            fmax_0 = fmax

        cost_mat = np.diag([self.Qx, self.Qv, self.Ru, self.Rf])
        final_cost_mat = np.diag([self.Qx, self.Qv])
        x_star = np.array([self.xt, self.vt, 0, 0])
        final_x_star = np.array([self.xt, self.vt])

        v0 = gcs.AddVertex(
            HPolyhedron.MakeBox(
                [self.x0, self.v0, -umax, 0], [self.x0, self.v0, umax, fmax_0]
            ),
            "v_0",
        )
        v0.AddCost((v0.x() - x_star).dot(cost_mat).dot(v0.x() - x_star))
        vertices["v_0"] = v0

        vN = gcs.AddVertex(
            HPolyhedron.MakeBox([0, -vmax], [xmax, vmax]), "v_" + str(self.N)
        )
        vN.AddCost(
            (vN.x() - final_x_star).dot(final_cost_mat).dot(vN.x() - final_x_star)
        )
        vertices["v_" + str(self.N)] = vN

        for n in range(1, self.N):
            # in contact, position is 0, force is positive
            # THIS IS WRONG, fix me
            vc_n = gcs.AddVertex(
                HPolyhedron.MakeBox([0, -vmax, -umax, 0], [xmax, vmax, umax, fmax]),
                "vc_" + str(n),
            )
            vc_n.AddCost((vc_n.x() - x_star).dot(cost_mat).dot(vc_n.x() - x_star))
            vertices["v_" + str(n)] = vc_n

        def add_cons_no_force(e: GraphOfConvexSets.Edge):
            h, m = self.h, self.m
            e.AddConstraint(e.xv()[0] == e.xu()[0] + h * e.xv()[1])
            e.AddConstraint(e.xv()[1] == e.xu()[1] + h / m * (e.xu()[2] + e.xu()[3]))
            e.AddConstraint(e.xu()[3] == 0)

        def add_cons_in_contact(e: GraphOfConvexSets.Edge):
            h, m = self.h, self.m
            e.AddConstraint(e.xv()[0] == e.xu()[0] + h * e.xv()[1])
            e.AddConstraint(e.xv()[1] == e.xu()[1] + h / m * (e.xu()[2] + e.xu()[3]))
            e.AddConstraint(e.xv()[0] == 0)

        for k in range(0, self.N):
            n = str(k)
            n1 = str(k + 1)
            ec = gcs.AddEdge(
                vertices["v_" + n], vertices["v_" + n1], "c v_" + n + "->v_" + n1
            )
            add_cons_in_contact(ec)
            en = gcs.AddEdge(
                vertices["v_" + n], vertices["v_" + n1], "n v_" + n + "->v_" + n1
            )
            add_cons_no_force(en)

        return gcs, v0, vN

    def solve_cr_with_gcs(self) -> float:
        gcs, v0, vN = self.build_gcs()
        options = GraphOfConvexSetsOptions()
        options.convex_relaxation = True
        options.max_rounded_paths = 0
        timer = timeit()
        solution = gcs.SolveShortestPath(v0, vN, options)
        timer.dt("GCS CR", verbose=self.verbose)
        diditwork(solution, verbose=self.verbose)
        INFO("-------------------", verbose=self.verbose)
        return solution.get_optimal_cost()

    def get_trajecories_from_edge_trajectory(
        self, solution: MathematicalProgramResult, edge_traj: npt.NDArray
    ):
        vertex_traj = []
        for edge in edge_traj:
            vertex_traj.append(solution.GetSolution(edge.u().x()))
        vertex_traj.append(solution.GetSolution(edge_traj[-1].v().x()))

        x_traj = np.array([x[0] for x in vertex_traj])
        v_traj = np.array([x[1] for x in vertex_traj])
        u_traj = np.array([x[2] for x in vertex_traj[:-1]])
        f_traj = np.array([x[3] for x in vertex_traj[:-1]])
        lcp_v_traj = self.get_lcp_violation(x_traj, f_traj)
        return x_traj, v_traj, f_traj, u_traj, lcp_v_traj

    def solve_micp_with_gcs(
        self,
    ) -> T.Tuple[
        float, float, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray
    ]:
        gcs, v0, vN = self.build_gcs()
        options = GraphOfConvexSetsOptions()
        options.convex_relaxation = False
        timer = timeit()
        solution = gcs.SolveShortestPath(v0, vN, options)
        solve_time = timer.dt("GCS MICP", verbose=self.verbose)
        diditwork(solution, verbose=self.verbose)
        INFO("-------------------", verbose=self.verbose)
        edge_traj = gcs.GetSolutionPath(v0, vN, solution)
        (
            x_traj,
            v_traj,
            u_traj,
            f_traj,
            lcp_v_traj,
        ) = self.get_trajecories_from_edge_trajectory(solution, edge_traj)
        return (
            solve_time,
            solution.get_optimal_cost(),
            x_traj,
            v_traj,
            u_traj,
            f_traj,
            lcp_v_traj,
        )

    def solve_bad_qcqp_convex_diff_decomposition(
        self,
    ) -> T.Tuple[
        float, float, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray
    ]:
        def add_socp_constraint(
            prog: MathematicalProgram, x: npt.NDArray, y: npt.NDArray, cost: float, n
        ):
            # add SDP relaxation of constraint x^T y = 0
            bar_p = prog.NewContinuousVariables(1, "bar_p" + str(n))[0]
            bar_q = prog.NewContinuousVariables(1, "bar_q" + str(n))[0]
            p = x + y
            q = x - y
            prog.AddLorentzConeConstraint(np.hstack((bar_p, p)))
            prog.AddLorentzConeConstraint(np.hstack((bar_q, q)))
            prog.AddLinearConstraint(bar_p == bar_q)
            prog.AddLinearConstraint(bar_p >= 0)

            cost_expr = cost * (bar_p**2 + bar_q**2)
            prog.AddQuadraticCost(cost_expr)
            return cost_expr

        prog = MathematicalProgram()
        x = np.hstack(([self.x0], prog.NewContinuousVariables(self.N, "x")))
        v = np.hstack(([self.v0], prog.NewContinuousVariables(self.N, "v")))
        f = prog.NewContinuousVariables(self.N, "f")
        u = prog.NewContinuousVariables(self.N, "u")

        # cost at 0 point for completeness
        total_cost = Expression(0)
        total_cost += self.Qx * (x[0] - self.xt) ** 2
        total_cost += self.Qv * (v[0] - self.vt) ** 2
        prog.AddQuadraticCost(total_cost)

        adjusted_cost = Expression(0)

        for n in range(self.N):
            # constraints
            prog.AddLinearConstraint(x[n + 1] >= 0)
            prog.AddLinearConstraint(f[n] >= 0)
            prog.AddLinearConstraint(x[n + 1] == x[n] + self.h * v[n + 1])
            prog.AddLinearConstraint(v[n + 1] == v[n] + self.h / self.m * (u[n] + f[n]))

            adjusted_cost += add_socp_constraint(
                prog, x[n + 1], f[n], self.qcqp_scaling_factor, n
            )

            # quadratic cost on each term
            total_cost = Expression(0)
            total_cost += self.Qx * (x[n + 1] - self.xt) ** 2
            total_cost += self.Qv * (v[n + 1] - self.vt) ** 2
            total_cost += self.Ru * (u[n] - 0) ** 2
            total_cost += self.Rf * (f[n] - 0) ** 2
            prog.AddQuadraticCost(total_cost)

        timer = timeit()
        solver = MosekSolver()
        solution = solver.Solve(prog)  # type: MathematicalProgramResult
        solve_time = timer.dt("convex decomposition QCQP", verbose=self.verbose)
        diditwork(solution, verbose=self.verbose)
        WARN(
            solution.get_optimal_cost() - solution.GetSolution(adjusted_cost),
            verbose=self.verbose,
        )
        INFO("-------------------", verbose=self.verbose)

        cost = (
            solution.get_optimal_cost() - solution.GetSolution(adjusted_cost).Evaluate()
        )
        ev = np.vectorize(lambda a: a.Evaluate())
        x_traj = ev(solution.GetSolution(x).reshape(self.N + 1))
        v_traj = ev(solution.GetSolution(v).reshape(self.N + 1))
        f_traj = solution.GetSolution(f).reshape(self.N)
        u_traj = solution.GetSolution(u).reshape(self.N)
        lcp_v_traj = self.get_lcp_violation(x_traj, f_traj)
        return solve_time, cost, x_traj, v_traj, f_traj, u_traj, lcp_v_traj

    def solve_bad_sdp_convex_diff_decomposition(
        self,
    ) -> T.Tuple[
        float, float, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray
    ]:
        def add_socp_constraint(
            prog: MathematicalProgram, x: npt.NDArray, y: npt.NDArray, cost: float, n
        ):
            # add SDP relaxation of constraint x^T y = 0
            P = prog.NewSymmetricContinuousVariables(1, "P")
            Q = prog.NewSymmetricContinuousVariables(1, "Q")
            p = x + y
            q = x - y

            P_full = np.vstack((np.hstack((P[0], p)), np.hstack((p, 1))))
            Q_full = np.vstack((np.hstack((Q[0], q)), np.hstack((q, 1))))
            prog.AddPositiveSemidefiniteConstraint(P_full)
            prog.AddPositiveSemidefiniteConstraint(Q_full)
            prog.AddLinearConstraint(np.trace(P) == np.trace(Q))

            cost_expr = cost * (np.trace(P) + np.trace(Q))
            prog.AddLinearCost(cost_expr)
            return cost_expr

        prog = MathematicalProgram()
        x = np.hstack(([self.x0], prog.NewContinuousVariables(self.N, "x")))
        v = np.hstack(([self.v0], prog.NewContinuousVariables(self.N, "v")))
        f = prog.NewContinuousVariables(self.N, "f")
        u = prog.NewContinuousVariables(self.N, "u")

        # cost at 0 point for completeness
        total_cost = Expression(0)
        total_cost += self.Qx * (x[0] - self.xt) ** 2
        total_cost += self.Qv * (v[0] - self.vt) ** 2
        prog.AddQuadraticCost(total_cost)

        adjusted_cost = Expression(0)

        for n in range(self.N):
            # constraints
            prog.AddLinearConstraint(x[n + 1] >= 0)
            prog.AddLinearConstraint(f[n] >= 0)
            prog.AddLinearConstraint(x[n + 1] == x[n] + self.h * v[n + 1])
            prog.AddLinearConstraint(v[n + 1] == v[n] + self.h / self.m * (u[n] + f[n]))

            adjusted_cost += add_socp_constraint(
                prog, x[n + 1], f[n], self.qcqp_scaling_factor, n
            )

            # quadratic cost on each term
            total_cost = Expression(0)
            total_cost += self.Qx * (x[n + 1] - self.xt) ** 2
            total_cost += self.Qv * (v[n + 1] - self.vt) ** 2
            total_cost += self.Ru * (u[n] - 0) ** 2
            total_cost += self.Rf * (f[n] - 0) ** 2
            prog.AddQuadraticCost(total_cost)

        timer = timeit()
        solver = MosekSolver()
        solution = solver.Solve(prog)  # type: MathematicalProgramResult
        solve_time = timer.dt("convex decomposition SDP", verbose=self.verbose)
        diditwork(solution, verbose=self.verbose)
        WARN(
            solution.get_optimal_cost() - solution.GetSolution(adjusted_cost),
            verbose=self.verbose,
        )
        INFO("-------------------", verbose=self.verbose)

        cost = (
            solution.get_optimal_cost() - solution.GetSolution(adjusted_cost).Evaluate()
        )
        ev = np.vectorize(lambda a: a.Evaluate())
        x_traj = ev(solution.GetSolution(x).reshape(self.N + 1))
        v_traj = ev(solution.GetSolution(v).reshape(self.N + 1))
        f_traj = solution.GetSolution(f).reshape(self.N)
        u_traj = solution.GetSolution(u).reshape(self.N)
        lcp_v_traj = self.get_lcp_violation(x_traj, f_traj)
        return solve_time, cost, x_traj, v_traj, f_traj, u_traj, lcp_v_traj

    def solve_our_qcqp_convex_diff_decomposition(
        self,
    ) -> T.Tuple[
        float, float, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray
    ]:
        def add_socp_constraint(
            prog: MathematicalProgram, x: npt.NDArray, y: npt.NDArray
        ):
            alpha, beta = np.sqrt(self.Qx), np.sqrt(self.Rf)
            p = alpha * x + beta * y
            q = alpha * x - beta * y
            r = prog.NewContinuousVariables(1)[0]
            prog.AddLinearConstraint(r >= 0)
            prog.AddLinearConstraint(p + q >= 0)
            prog.AddLinearConstraint(p - q >= 0)
            prog.AddRotatedLorentzConeConstraint(r, Expression(1), p * p, tol=1e-10)
            prog.AddRotatedLorentzConeConstraint(r, Expression(1), q * q, tol=1e-10)
            prog.AddLinearCost(r)

        prog = MathematicalProgram()
        x = np.hstack(([self.x0], prog.NewContinuousVariables(self.N, "x")))
        v = np.hstack(([self.v0], prog.NewContinuousVariables(self.N, "v")))
        f = prog.NewContinuousVariables(self.N, "f")
        u = prog.NewContinuousVariables(self.N, "u")

        # cost at 0 point for completeness
        total_cost = Expression(0)
        total_cost += self.Qx * (x[0] - self.xt) ** 2
        total_cost += self.Qv * (v[0] - self.vt) ** 2
        prog.AddQuadraticCost(total_cost)

        for n in range(self.N):
            # constraints
            prog.AddLinearConstraint(x[n + 1] >= 0)
            prog.AddLinearConstraint(f[n] >= 0)
            prog.AddLinearConstraint(x[n + 1] == x[n] + self.h * v[n + 1])
            prog.AddLinearConstraint(v[n + 1] == v[n] + self.h / self.m * (u[n] + f[n]))

            add_socp_constraint(prog, x[n + 1], f[n])

            # quadratic cost on each term
            total_cost = Expression(0)
            total_cost += self.Qx * (self.xt) ** 2 - 2 * self.Qx * x[n + 1] * self.xt
            total_cost += self.Qv * (v[n + 1] - self.vt) ** 2
            total_cost += self.Ru * (u[n] - 0) ** 2
            prog.AddQuadraticCost(total_cost)

        timer = timeit()
        solver = MosekSolver()
        solution = solver.Solve(prog)  # type: MathematicalProgramResult
        solve_time = timer.dt("our convex decomposition QCQP", verbose=self.verbose)
        diditwork(solution, verbose=self.verbose)
        WARN(solution.get_optimal_cost(), verbose=self.verbose)
        INFO("-------------------", verbose=self.verbose)

        cost = solution.get_optimal_cost()
        ev = np.vectorize(lambda a: a.Evaluate())
        x_traj = ev(solution.GetSolution(x).reshape(self.N + 1))
        v_traj = ev(solution.GetSolution(v).reshape(self.N + 1))
        f_traj = solution.GetSolution(f).reshape(self.N)
        u_traj = solution.GetSolution(u).reshape(self.N)
        lcp_v_traj = self.get_lcp_violation(x_traj, f_traj)
        return solve_time, cost, x_traj, v_traj, f_traj, u_traj, lcp_v_traj

    def solve_our_qcqp_convex_diff_decomposition_with_warmstart(
        self,
    ) -> T.Tuple[
        float, float, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray
    ]:
        (
            solve_time,
            cost,
            x_traj,
            v_traj,
            f_traj,
            u_traj,
            lcp_v_traj,
        ) = self.solve_our_qcqp_convex_diff_decomposition()
        (
            solve_time2,
            cost,
            x_traj,
            v_traj,
            f_traj,
            u_traj,
            lcp_v_traj,
        ) = self.warmstart_with_snopt(x_traj, v_traj, f_traj, u_traj)
        return (
            solve_time + solve_time2,
            cost,
            x_traj,
            v_traj,
            f_traj,
            u_traj,
            lcp_v_traj,
        )

    def solve_bad_qcqp_convex_diff_decomposition_with_warmstart(
        self,
    ) -> T.Tuple[
        float, float, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray
    ]:
        (
            solve_time,
            cost,
            x_traj,
            v_traj,
            f_traj,
            u_traj,
            lcp_v_traj,
        ) = self.solve_bad_qcqp_convex_diff_decomposition()
        (
            solve_time2,
            cost,
            x_traj,
            v_traj,
            f_traj,
            u_traj,
            lcp_v_traj,
        ) = self.warmstart_with_snopt(x_traj, v_traj, f_traj, u_traj)
        return (
            solve_time + solve_time2,
            cost,
            x_traj,
            v_traj,
            f_traj,
            u_traj,
            lcp_v_traj,
        )


def run_a_test_for_suboptimality_gap(Qx=1, Rf=0.01, qcqp_scaling_factor=0.1):
    sdp_violations = []
    qcqp_violations = []
    sdp_times = []
    qcqp_times = []
    for v0 in range(-15, 15):
        cart = ContactCart(
            v0=v0,
            qcqp_scaling_factor=qcqp_scaling_factor,
            verbose=False,
            Qx=Qx,
            Rf=Rf,
            umax=50,
        )

        (
            chordal_time,
            chordal_sdp_cost,
            _,
            _,
            _,
            _,
            _,
        ) = cart.solve_chordal_nonconvex_qcqp_sdp_relaxation()
        # chordal_time, chordal_sdp_cost, _, _, _, _, _ = cart.solve_our_qcqp_convex_diff_decomposition_with_warmstart()
        # _, micp_cost, _, _, _, _, _ = cart.solve_micp_with_gcs()
        (
            _,
            micp_cost,
            _,
            _,
            _,
            _,
            _,
        ) = cart.solve_our_qcqp_convex_diff_decomposition_with_warmstart()
        # qcqp_time, qcqp_ipopt_cost, _, _, _, _, _ = cart.solve_our_qcqp_convex_diff_decomposition_with_warmstart()
        # qcqp_time, qcqp_ipopt_cost, _, _, _, _, _ = cart.solve_bad_qcqp_convex_diff_decomposition_with_warmstart()
        qcqp_time, qcqp_ipopt_cost, _, _, _, _, _ = cart.solve_with_snopt()
        sdp_violations.append(chordal_sdp_cost / micp_cost)
        qcqp_violations.append(qcqp_ipopt_cost / micp_cost)
        sdp_times.append(chordal_time)
        qcqp_times.append(qcqp_time)

    print("SDP", np.min(sdp_violations), np.max(sdp_violations), np.mean(sdp_times))
    # print("QCQP SNOPT", np.min(qcqp_violations), np.max(qcqp_violations), np.mean(qcqp_violations), np.mean(qcqp_times))
    print(
        "their QCQP SNOPT",
        np.min(qcqp_violations),
        np.max(qcqp_violations),
        np.mean(qcqp_violations),
        np.mean(qcqp_times),
    )
    print(np.round(qcqp_violations, 3))


if __name__ == "__main__":
    cart = ContactCart(Qx=1, Rf=1)
    # cart.solve_with_ipopt()
    # cart.solve_nonconvex_qcqp_sdp_relaxation()
    # cart.solve_cr_with_gcs()
    # cart.solve_micp_with_gcs()
    # cart.solve_bad_qcqp_convex_diff_decomposition()
    # cart.solve_bad_sdp_convex_diff_decomposition()
    # cart.solve_our_qcqp_convex_diff_decomposition()
    # cart.solve_chordal_nonconvex_qcqp_sdp_relaxation()
    # cart.solve_our_qcqp_convex_diff_decomposition_with_warmstart()
    run_a_test_for_suboptimality_gap(qcqp_scaling_factor=1)
