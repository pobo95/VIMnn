# Truncated OT-weighted probability-density multipoles v1

## Operating domain

The feature integration consumes a converged `TRAIN_FIXED` result; it never
solves or repairs OT.  The synthetic acceptance domain is recorded as
`epsilon_OT=0.5`, `ell_OT=1.5`, `float64`, 256 fixed log-Sinkhorn iterations,
the cost `MIC squared distance / (2 ell_OT^2)`, marginal infinity-norm
tolerance `1e-7`, and solver contract `dense_aggregate_vacancy_ot_v1`.
Every accepted structure must have finite values, `rho <= 1e-7`,
`0 <= q_s <= 1`, and vacancy-mass error at most the same tolerance.  This is
not a claim about an as-yet unavailable dataset.  The existing
`epsilon_OT=0.02` extreme-cost fixture remains outside the supported fixed
training domain.  There is no post-normalization, residual clipping, or
reconstruction of `q` from row sums.

## Definition and meaning

For the reference-site-to-atom MIC vector `d_si`, let `y_si=d_si/ell_feature`
and `xi_si=y_si.y_si`.  For species alpha,

```text
rho_s^alpha(y) = sum_i P_si 1[Z_i=alpha] delta(y-y_si)
A_s,alpha,n,l,m = sum_i P_si 1[Z_i=alpha] b_nl(xi_si) S_lm(y_si).
```

These are **truncated OT-weighted probability-density multipoles**.  Only the
separate exact constant `l=0` occupancy channel is probability mass.  Higher
components may be negative.  With `lmax=2` and a finite radial basis the
representation is incomplete; distinct atomic mixtures can share its
coefficients, and no injectivity for the full atomic structure is claimed.
The vacancy field is the original `OTResult.q`, preserved as an immutable raw
`0e` scalar alongside exact species occupancies.

## Solid harmonics and radial basis

The regular Cartesian solid harmonics use e3nn 0.4.4 with
`Irreps.spherical_harmonics(2)`, `normalize=False`, and
`normalization="component"`.  No unit direction, square root, norm, or extra
`r^l` factor is formed.  Thus `S_00(0)=1` and all `l>0` components vanish at
the origin.

For `u=xi/(r_cut/ell_feature)^2`, compact channels are
`u^n(1-10u^3+15u^4-6u^5)` inside the cutoff and exactly zero outside.  The
value and first two derivatives match zero at the cutoff.  Exact occupancy
channels have no radial function or cutoff.

## Fixed layout

The flat output order is: species exact occupancy `0e`; compact `l=0`;
compact `l=1`; compact `l=2`.  Within each compact block the order is species,
radial index, then e3nn component (fastest).  Public metadata records all
slices, parity, radial index, species order, scale/cutoff, basis/layout
versions, normalization, and displacement orientation.

## MIC domain

Features use the certified triclinic MIC vector itself.  Nearest and
second-nearest image distances and their gap are diagnostics only and never
threshold `P` or change a feature.  Smooth force/stress claims apply only on
an accepted open domain with a stable, unique MIC branch.  No global
smoothness is claimed at MIC ties or the periodic cut locus.
