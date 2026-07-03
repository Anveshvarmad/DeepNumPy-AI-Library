import json
import numpy as np


def save_checkpoint(path, model, epoch=0, history=None, extra=None):
    state = model.state_dict()

    metadata = {
        "epoch": epoch,
        "history": history if history is not None else {},
        "extra": extra if extra is not None else {}
    }

    payload = {
        key: value
        for key, value in state.items()
    }

    payload["__metadata__"] = np.array(json.dumps(metadata))

    np.savez(path, **payload)


def load_checkpoint(path, model):
    loaded = np.load(path, allow_pickle=True)

    state = {}

    for key in loaded.files:
        if key == "__metadata__":
            continue

        state[key] = loaded[key]

    model.load_state_dict(state)

    if "__metadata__" in loaded.files:
        return json.loads(loaded["__metadata__"].item())

    return {}
