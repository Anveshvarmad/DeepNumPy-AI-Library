import numpy as np
import ai


def test_count_parameters():
    model = ai.nn.Sequential(
        ai.nn.Linear(2, 4),
        ai.nn.ReLU(),
        ai.nn.Linear(4, 1)
    )

    total = ai.utils.count_parameters(model)

    assert total == 17


def test_model_summary_contains_layers():
    model = ai.nn.Sequential(
        ai.nn.Linear(2, 4),
        ai.nn.ReLU(),
        ai.nn.Linear(4, 1)
    )

    summary = ai.utils.model_summary(model)

    assert "Sequential" in summary
    assert "Linear" in summary
    assert "Total parameters" in summary


def test_trace_graph_nodes_and_edges():
    x = ai.Tensor(np.array([[1.0, 2.0]]), requires_grad=True, name="x")
    W = ai.Parameter(np.array([[3.0], [4.0]]), name="W")

    y = x @ W
    loss = y.mean()

    nodes, edges = ai.utils.trace_graph(loss)

    assert len(nodes) >= 4
    assert len(edges) >= 3


def test_build_dot_contains_ops():
    x = ai.Tensor(np.array([[1.0, 2.0]]), requires_grad=True, name="x")
    W = ai.Parameter(np.array([[3.0], [4.0]]), name="W")

    y = x @ W
    loss = y.mean()

    dot = ai.utils.build_dot(loss)

    assert "digraph" in dot
    assert "matmul" in dot
    assert "mean" in dot


def test_save_graph(tmp_path):
    x = ai.Tensor(np.array([1.0, 2.0]), requires_grad=True, name="x")
    y = (x * x).sum()

    path = tmp_path / "graph_output"

    saved_path = ai.utils.save_graph(y, filename=path)

    assert saved_path.endswith(".dot")
    assert path.with_suffix(".dot").exists()
