"""Disjoint batching of fixed reference graphs."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from .topology import ReferenceGraphTopology

@dataclass(frozen=True)
class BatchedReferenceGraph:
    edge_index: torch.Tensor
    shifts: torch.Tensor
    node_batch: torch.Tensor
    edge_batch: torch.Tensor
    node_offsets: tuple[int,...]


def batch_reference_graphs(topologies: list[ReferenceGraphTopology])->BatchedReferenceGraph:
    if not topologies: raise ValueError("at least one topology is required")
    device=topologies[0].edge_index.device
    if any(t.edge_index.device!=device for t in topologies): raise ValueError("all topologies must share device")
    edges=[]; shifts=[]; node_batch=[]; edge_batch=[]; offsets=[]; offset=0
    for b,t in enumerate(topologies):
        offsets.append(offset); edges.append(t.edge_index+offset); shifts.append(t.shifts)
        node_batch.append(torch.full((t.num_sites,),b,dtype=torch.long,device=device))
        edge_batch.append(torch.full((t.num_edges,),b,dtype=torch.long,device=device)); offset+=t.num_sites
    return BatchedReferenceGraph(torch.cat(edges,dim=1),torch.cat(shifts),torch.cat(node_batch),torch.cat(edge_batch),tuple(offsets))
