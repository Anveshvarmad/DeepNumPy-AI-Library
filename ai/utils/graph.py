from pathlib import Path


def _escape(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _shape_str(tensor):
    if tensor.data.shape == ():
        return "scalar"
    return str(tensor.data.shape)


def trace_graph(output_tensor):
    nodes = set()
    edges = set()

    def build(tensor):
        if tensor not in nodes:
            nodes.add(tensor)

            for child in tensor._prev:
                edges.add((child, tensor))
                build(child)

    build(output_tensor)

    return nodes, edges


def build_dot(output_tensor, graph_name="ComputationGraph"):
    nodes, edges = trace_graph(output_tensor)

    lines = [
        f'digraph "{_escape(graph_name)}" {{',
        "  rankdir=LR;",
        "  graph [fontsize=10];",
        "  node [shape=record, fontsize=10];",
        "  edge [fontsize=9];",
    ]

    for tensor in nodes:
        tensor_id = f"tensor_{id(tensor)}"

        name = tensor.name if tensor.name is not None else "Tensor"
        op = tensor._op if tensor._op else "leaf"
        shape = _shape_str(tensor)
        requires_grad = tensor.requires_grad

        label = (
            f"{_escape(name)} | "
            f"shape={_escape(shape)} | "
            f"op={_escape(op)} | "
            f"grad={requires_grad}"
        )

        if tensor.requires_grad:
            lines.append(f'  {tensor_id} [label="{label}", style="filled"];')
        else:
            lines.append(f'  {tensor_id} [label="{label}"];')

    op_index = 0

    for source, target in edges:
        source_id = f"tensor_{id(source)}"
        target_id = f"tensor_{id(target)}"

        if target._op:
            op_id = f"op_{op_index}"
            op_index += 1

            lines.append(
                f'  {op_id} [label="{_escape(target._op)}", '
                f'shape=oval, style="dashed"];'
            )
            lines.append(f"  {source_id} -> {op_id};")
            lines.append(f"  {op_id} -> {target_id};")
        else:
            lines.append(f"  {source_id} -> {target_id};")

    lines.append("}")

    return "\n".join(lines)


def save_graph(output_tensor, filename="computation_graph", graph_name="ComputationGraph"):
    path = Path(filename)

    if path.suffix != ".dot":
        path = path.with_suffix(".dot")

    path.parent.mkdir(parents=True, exist_ok=True)

    dot = build_dot(output_tensor, graph_name=graph_name)
    path.write_text(dot)

    return str(path)


def render_graph(output_tensor, filename="computation_graph", graph_name="ComputationGraph", format="png"):
    try:
        from graphviz import Source
    except ImportError as exc:
        raise ImportError("Install graphviz Python package to render graphs") from exc

    dot = build_dot(output_tensor, graph_name=graph_name)
    source = Source(dot)

    output_path = source.render(
        filename=filename,
        format=format,
        cleanup=True
    )

    return output_path
