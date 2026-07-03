import numpy as np

from ..core import Parameter
from ..nn.modules import Module


def _numel(parameter):
    return int(np.prod(parameter.shape))


def count_parameters(model_or_parameters, trainable_only=True):
    if hasattr(model_or_parameters, "parameters"):
        parameters = model_or_parameters.parameters()
    else:
        parameters = list(model_or_parameters)

    total = 0
    seen = set()

    for parameter in parameters:
        if id(parameter) in seen:
            continue

        seen.add(id(parameter))

        if trainable_only and not parameter.requires_grad:
            continue

        total += _numel(parameter)

    return total


def _direct_parameters(module):
    params = []

    for name, value in module.__dict__.items():
        if isinstance(value, Parameter):
            params.append((name, value))

    return params


def _child_modules(module):
    children = []

    for name, value in module.__dict__.items():
        if isinstance(value, Module):
            children.append((name, value))

        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                if isinstance(item, Module):
                    children.append((f"{name}.{index}", item))

        elif isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, Module):
                    children.append((f"{name}.{key}", item))

    return children


def model_summary(model):
    rows = []
    seen = set()

    def walk(module, name, depth):
        if id(module) in seen:
            return

        seen.add(id(module))

        direct_param_count = sum(_numel(parameter) for _, parameter in _direct_parameters(module))
        total_param_count = count_parameters(module, trainable_only=False)
        trainable_param_count = count_parameters(module, trainable_only=True)

        rows.append({
            "name": name,
            "type": module.__class__.__name__,
            "depth": depth,
            "direct": direct_param_count,
            "total": total_param_count,
            "trainable": trainable_param_count,
        })

        for child_name, child in _child_modules(module):
            display_name = f"{name}.{child_name}" if name else child_name
            walk(child, display_name, depth + 1)

    walk(model, "model", 0)

    header = (
        f"{'Layer':<35} "
        f"{'Type':<28} "
        f"{'Direct Params':>14} "
        f"{'Total Params':>14} "
        f"{'Trainable':>12}"
    )

    divider = "-" * len(header)
    lines = [header, divider]

    for row in rows:
        indent = "  " * row["depth"]
        layer_name = f"{indent}{row['name']}"

        lines.append(
            f"{layer_name:<35} "
            f"{row['type']:<28} "
            f"{row['direct']:>14} "
            f"{row['total']:>14} "
            f"{row['trainable']:>12}"
        )

    lines.append(divider)
    lines.append(f"Total parameters: {count_parameters(model, trainable_only=False)}")
    lines.append(f"Trainable parameters: {count_parameters(model, trainable_only=True)}")

    return "\n".join(lines)


def print_model_summary(model):
    summary = model_summary(model)
    print(summary)
    return summary
