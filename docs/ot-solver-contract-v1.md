# Dense Aggregate-Vacancy OT Solver Contract v1

## Scope

Version 1 implements dense balanced entropic transport from M reference sites
to N actual atoms plus, only when K=M-N>0, one aggregate vacancy reservoir.
N>M fails fast. The OT objective is representation machinery and is never
added to physical energy or training loss. epsilon_OT and ell_OT are fixed,
finite, positive hyperparameters.

For Cartesian row vectors:

    d_si = MIC_H(r_i - R_s)
    C_si = (d_si . d_si) / (2 ell_OT^2)
    C_sV = 0

The augmented row and column marginals are a=1_M and b=(1_N,K). The
primal objective is

    <Gamma,C_bar> + epsilon_OT sum_sj Gamma_sj(log Gamma_sj - 1).

The triclinic MIC search starts from the component-rounded fractional image,
then uses the smallest singular value of H and that candidate distance to
certify a finite integer search radius. Every lattice shift that can improve
the candidate lies inside this radius. Candidate selection is locally
piecewise differentiable away from MIC ties; the selected Cartesian arithmetic
remains in the autograd graph.

P is Gamma[:,:N]. For K>0, q is the vacancy column itself. For K=0 no
vacancy column exists and q is a separately allocated exact zero tensor. The
implementation never reconstructs q as 1-P1, clips a plan, or normalizes
marginals after a solve. N=0 is the analytic P[M,0], q=1_M case.

K identical unit vacancy columns have Gamma_(s,V_k)=q_s/K at the converged
symmetric optimum and

    J_expanded = J_aggregate - epsilon_OT K log K.

## Dual and gauge

    Gamma_sj = exp((f_s+g_j-C_bar_sj)/epsilon_OT)
    F = [Gamma 1-a, Gamma^T 1-b]
    z = [1_M,-1_J]
    Pi(x) = x-z(z^T x)/(M+J).

Every full Sinkhorn update and accepted Newton update creates new projected
dual tensors. The production Newton operator is matrix-free:

    J[u,v] = [W1,W^T1],  W_sj=Gamma_sj(u_s+v_j)/epsilon_OT
    A(x) = Pi J Pi x + rho_gauge (I-Pi)x.

Projected PCG uses the current row/column Jacobian diagonal as a Jacobi
preconditioner and projects the preconditioned residual and every search
direction. Dense Jacobians are permitted only in tests.

## TRAIN_FIXED

The production force-training path is fixed-unrolled log-Sinkhorn with
deterministic zero duals. It has a fixed iteration count and no data-dependent
early stop, Python break, line search, fallback, detach, no_grad, data access,
in-place dual update, custom autograd Function, or activation checkpoint. It
does not form exp(-C/epsilon). Diagnostics do not control arithmetic. The
entire selected graph supports cost and position gradcheck and gradgradcheck.

Newton/PCG is not exposed as a TRAIN_FIXED production solver in v1.

## EVAL_ADAPTIVE

The evaluation-only solvers are sinkhorn, newton_krylov, and hybrid. Hybrid
uses a fixed log-Sinkhorn warm start, gauge recentering, adaptive Newton-PCG,
and dual-objective Armijo backtracking. A PCG, line-search, or Newton failure
causes an explicit additional log-Sinkhorn fallback and is recorded. There is
no silent solver substitution: fallback belongs only to the requested hybrid
algorithm. Standalone newton_krylov fails fast.

Adaptive residual and branch decisions may inspect detached scalars, but the
accepted candidate arithmetic is not detached. Force differentiation is local
to a stable selected arithmetic branch. EVAL_ADAPTIVE is not a training-loss
path.

## Ragged batches and diagnostics

A ragged batch is a sequence of independent problem solves. Every graph has
its own atom columns, aggregate vacancy reservoir, dual gauge, convergence
status, and diagnostics. Problems are never padded or coupled.

OTResult returns Gamma, P, q, f, g, row/column maximum residuals, convergence,
Sinkhorn/Newton/PCG counts, line-search reductions, fallback status, solver
name, and path name. Small-epsilon diagnostics also report plan underflow and
gradient conditioning in tests.
