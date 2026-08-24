"""Report parameter/dimension scaling without allocating large graphs."""
import json
from refsite_mlip.interactions import CentralConditionedHigherBody,HigherBodyConfig

def make(n,mode="uuu"):
    return CentralConditionedHigherBody(HigherBodyConfig("2x0e+4x0e+4x1o+4x2e",2,2,2,n,2,3,(8,),6.,3.,1.,mode))

def main():
    rows=[]
    for n in (2,4,8,16):
        m=make(n); row={"n_corr":n,**m.parameter_diagnostics(),"edge_weight_numel":m.edge_density.edge_tp.weight_numel,"Z1_dim":m.irreps_Z1.dim,"Z2_dim":m.irreps_Z2.dim,"Z3_dim":m.irreps_Z3.dim,"peak_node_dim":max(m.irreps_Z1.dim,m.irreps_C1.dim,m.irreps_source.dim)}; rows.append(row)
    dense=make(2,"uvw").parameter_diagnostics()
    print(json.dumps({"uuu":rows,"dense_uvw_n_corr_2":dense},indent=2))
if __name__=="__main__": main()
