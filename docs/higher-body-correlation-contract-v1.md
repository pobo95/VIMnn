# Reference-site higher-body correlation algebra v1

This milestone implements **Layer-local degree-1/2/3 reference-site
neighbor-density correlations with explicit central conditioning**.

A fixed directed periodic graph uses `source=t`, `target=s` and integer shifts
`n_st`, with edge vector `(S_t-S_s+n_st)H`. The zero-shift self edge is
excluded; nonzero self images and distinct periodic images are retained. The
canonical topology is built once at `cutoff+skin`; current-cell forwards only
recompute Cartesian vectors. A spectral strain certificate guarantees omitted
edges cannot enter the physical cutoff, otherwise execution fails.

The edge basis uses normalized e3nn spherical harmonics (`normalize=True`,
`normalization="component"`) and a squared-distance radial basis. A quintic C2
cutoff multiplies the radial-head output after its final affine layer. Edge
messages use external unshared `uvw` weights and aggregate at targets with a
fixed `1/sqrt(z_avg)` factor.

Raw state `[p^1,...,p^A,q]` is immutable. The scalar central conditioner is
`[1,p^1,...,p^A,q,e_T,q e_T]`; the same conditioner is directly present in
source edge features. `C1=A`, `C2=TP(C1,A)`, and `C3=TP(C2,A)` use all legal
triangle/parity paths, l truncation, shared internal `uuu` weights. Each
`Znu=TP(c_bar,Cnu)` is an exact parameter-free `uvuv` outer product, preserving
constant, chemistry, explicit q, site type, and q-times-site-type slices.

Nominally `c*C1`, `c*C2`, and `c*C3` contain center plus one, up to two, and up
to three neighbors. Repeated neighbor indices are included. Intermediate
angular truncation occurs; source states already contain OT probability
multipoles and P/q depend on a global constrained OT problem. This is neither
a strict atomic body-order expansion nor a complete/injective representation,
and is not called a strict four-body potential.

Production prototype correlations use `uuu`. Dense `uvw` correlation is only a
small `n_corr<=2` scaling reference. Gate, residual hidden updates, energy
readout, and training are outside this contract.
