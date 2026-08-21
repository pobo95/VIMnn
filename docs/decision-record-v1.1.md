# Reference-Site MLIP Decision Record v1.1

## Scope

The model maps actual atoms to a typed reference lattice, builds a probability
field by balanced entropic transport, and predicts energy from an equivariant
reference-site graph.  Vacancies are not atoms or graph nodes.  Version 1
requires `N <= M`; `K = M - N` is represented by one aggregate vacancy
reservoir.  The OT objective is not physical energy.  This milestone implements
only the typed reciprocal translation gauge.

All Cartesian vectors use row-vector convention.

## DR-01: aggregate vacancy transport

For aligned sites `R_s = o + (S_s + delta) H`,

```text
d_si = MIC_H(r_i - R_s)
C_si = ||d_si||^2 / (2 ell_OT^2)
C_sV = 0
```

With `a = 1_M`, `b = (1_N, K)`, and `K > 0`, the objective is

```text
J(Gamma) = <C_bar, Gamma>
         + epsilon_OT sum_sj Gamma_sj (log Gamma_sj - 1).
```

For identical vacancy costs, `K` unit vacancy marginals, and a converged OT
optimum, `Gamma_(s,V_k) = q_s/K` and

```text
J_expanded = J_aggregate - epsilon_OT K log K.
```

The difference is constant with respect to transport and geometry but not with
respect to `epsilon_OT`.  Therefore `epsilon_OT` is a fixed, non-trainable v1
hyperparameter, and neither OT objective is added to predicted physical energy
or training loss.  At `K = 0` the vacancy column is absent and `q` is exactly
zero.  `N > M` fails fast.

## DR-02: immutable raw site state and central conditioning

```text
p_s^alpha = sum_(i:Z_i=alpha) P_si
sum_alpha p_s^alpha + q_s = 1
c_s = [p_s^1, ..., p_s^A, q_s, e_type(T_s)].
```

`c_s` is an immutable collection of `0e` scalars retained beside every hidden
state.  Future edge messages directly retain source raw state:

```text
a_st^(k) = TP(
    h_t^(k) concat E_raw^(k) c_t,
    Y(R_st);
    w(r_st)
).
```

After neighbor-density correlations `C_s^(nu)`, TensorProduct instructions
must explicitly contain

```text
q_s(0e) x C_s^(nu)(l^p) -> l^p.
```

Every interaction has a raw scalar skip,

```text
h_(s,0e)^(k+1) = h_(s,0e)^(k+1,interaction) + W_c^(k) c_s,
```

and the final readout is

```text
epsilon_s = Readout(h_(s,0e)^L concat c_s).
```

Since `sum_s q_s = K`, a site-independent linear q-only energy contains no
vacancy-location information.  Localization must arise from `q_s C_s^(nu)`,
site type, chemistry, and nonlinear neighbor coupling.  The model is called a
“central-site-conditioned, layer-local up-to-4-body reference-site
density-correlation model”, not a strict atomic four-body potential.

## DR-03: typed reciprocal translation gauge

Coordinates and aligned sites are

```text
x_i = (r_i - o) H^-1
R_s = o + (S_s + delta) H.
```

For fixed integer modes `n_g`, fixed typed channel maps `u_c`, `v_c`, and
positive fixed weights,

```text
A_gc = sum_i u_c(Z_i) exp(2 pi i n_g . x_i)
B_gc = sum_s v_c(T_s) exp(2 pi i n_g . S_s)
C_g  = sum_c alpha_c A_gc B_gc*
A(delta) = sum_g lambda_g Re[C_g exp(-2 pi i n_g . delta)].
```

Principal phases are not unwrapped by ordinary least squares.  Translation is
defined on `T^3 / S_T`, where

```text
S_T = {tau: exists pi_tau,
       S_(pi_tau(s)) = S_s + tau mod Z^3 and
       T_(pi_tau(s)) = T_s}.
```

Primary modes are accepted only if their integer alias kernel

```text
K_primary = {tau: exp(2 pi i n_g . tau)=1 for every informative g}
```

equals `S_T`; rank three alone is insufficient.

### Training phase solver

Training uses fixed primary modes, a translation-covariant torus
initialization, fixed damped Newton iterations, a fixed step/damping schedule,
and batched 3x3 `torch.linalg.solve`.  Every iteration is unrolled.  The path
has no `.item()`, detach, in-place update, wrapping, early stopping, hard
candidate selection, line search, or adaptive branch.  A fixed zero
initialization is forbidden.

The periodic objective is evaluated at the unwrapped iterate:

```text
g(delta) = 2 pi sum_g lambda_g Im(D_g) n_g
H(delta) = -(2 pi)^2 sum_g lambda_g Re(D_g) n_g n_g^T
D_g = C_g exp(-2 pi i n_g . delta)
G_k = -H(delta_k) + mu_k I
Delta_k = solve(G_k, g_k)
delta_tilde_(k+1) = delta_tilde_k + eta_k Delta_k.
```

Canonical wrapping is used only for output, logging, and equivalence tests.
The solver is locally differentiable on the accepted open domain where typed
amplitudes, the non-equivalent phase gap, Hessian conditioning, and MIC branches
are stable.

### Evaluation

Evaluation candidates are covariant:

```text
delta_candidate_j = delta_initial + fixed_offset_j,
```

with offsets fixed on `T^3/S_T`.  Evaluation groups stabilizer-equivalent
candidates, checks the best/second-best non-equivalent gap, selects a stable
branch, and performs differentiable local refinement without detaching the
selected tensor.  Low amplitude, low gap, bad Hessian, or large final residual
fails fast; centroid fallback is forbidden.

Training and evaluation phases are equivalent iff

```text
[delta_train - delta_eval]_(T^3) in S_T.
```

At `delta + tau`, transport fields compare after the induced site permutation:

```text
P_si(delta + tau) = P_(pi_tau(s),i)(delta)
q_s(delta + tau)  = q_(pi_tau(s))(delta).
```

Reference graph indices and periodic shifts use the same permutation.  Energy,
forces, and stress are equal without output permutation.

Static acceptance checks non-extinct typed modes, integer alias/stabilizer
equality, runtime atomic/cross amplitudes, phase-Hessian curvature and
conditioning, the non-equivalent phase gap, and energy/force equivalence before
and after canonical phase wrapping. It also covers a nonzero origin and typed
sublattice aliases. The fixed damping schedule is accepted only when every
recorded regularized curvature satisfies the configured positive lower bound.

### Phase-solver verification contract

The training path must pass covariant-initialization, fixed finite-iteration
translation-covariance, gradcheck, and gradgradcheck tests. The evaluation path
must agree with training modulo the typed stabilizer on a stable branch and
must fail fast at candidate-switching boundaries, extinct/collapsed modes, an
insufficient non-equivalent phase gap, bad Hessian conditioning, or a large
final gradient residual. Regression fixtures include damping sensitivity,
nonzero origin, triclinic affine strain, lattice-vector translations, and typed
sublattice aliasing.

### Stress

For `F_eps = I + eps`, row-vector affine deformation is

```text
H_eps = H F_eps
r_eps = r F_eps
o_eps = o F_eps
sigma_ab = (1/V) dE(r F_eps, H F_eps, o F_eps)/d eps_ab |_(eps=0).
```

ASE sign and Voigt ordering are fixed by finite difference tests.

## DR-04: regular solid harmonics and radial smoothness

e3nn 0.4.4 uses

```text
normalize=False
normalization="component"
irreps_in="1o".
```

Solid harmonics are evaluated as Cartesian homogeneous polynomials without
forming `d/||d||`: `S_00(0)=1` and `S_lm(0)=0` for `l>0`.  Radial channels are

```text
B_nl(r) = b_nl((d.d)/ell_feature^2),
```

so `r^l` occurs only in the solid harmonic.  `b_nl` is at least C2.  The exact
occupancy channel has `B_00=1`, `S_00=1`, and no cutoff.  A compact channel
extended by zero at `r_c` satisfies `B(r_c)=B'(r_c)=B''(r_c)=0`.  General
`B(r)` additionally requires `B'(0)=0`.  Position-dependent networks and gates
use smooth SiLU/softplus/tanh-family activations, never ReLU.
