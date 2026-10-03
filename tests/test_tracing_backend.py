"""Run with AUTOEXPLAIN_TRACING_TEST=1 in the isolated tracing environment."""
import os
import pytest
import torch

pytestmark=pytest.mark.skipif(os.environ.get('AUTOEXPLAIN_TRACING_TEST')!='1',
                             reason='Requires isolated transformers-4 tracing environment')


def test_real_circuit_tracer():
    from autoexplain.circuits import tiny_tracing_model,top_graph_edges
    from autoexplain.frontier import trace_circuit
    torch.set_num_threads(2)
    graph=trace_circuit(tiny_tracing_model(),torch.tensor([1,3,4,5]),max_nodes=16)
    assert graph.adjacency_matrix.isfinite().all()
    edges=top_graph_edges(graph,k=5)
    assert len(edges)==5 and all(set(e)=={'source','target','effect'} for e in edges)
