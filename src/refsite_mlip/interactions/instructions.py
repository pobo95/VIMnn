"""Auditable e3nn TensorProduct instruction construction."""
from __future__ import annotations
from dataclasses import asdict,dataclass

@dataclass(frozen=True)
class InstructionMetadata:
    input1_block:int; input2_block:int; output_block:int; connection_mode:str; has_weight:bool
    def to_dict(self): return asdict(self)


def legal_instructions(irreps1,irreps2,irreps_out,mode:str,has_weight:bool):
    instructions=[]
    for i1,(_,ir1) in enumerate(irreps1):
        for i2,(_,ir2) in enumerate(irreps2):
            products=list(ir1*ir2)
            for io,(_,iro) in enumerate(irreps_out):
                if iro in products:
                    instructions.append((i1,i2,io,mode,has_weight))
    covered={i[2] for i in instructions}
    if covered!=set(range(len(irreps_out))):
        raise ValueError("at least one requested output irrep has no legal TensorProduct path")
    return instructions


def instruction_metadata(tp):
    return tuple(InstructionMetadata(i.i_in1,i.i_in2,i.i_out,i.connection_mode,i.has_weight) for i in tp.instructions)
