import torch

def average_state_dicts(state_dicts):
    """Compute the unweighted mean of client tensor dictionaries."""

    if not state_dicts:
        raise ValueError("At least one client update is required")

    reference_keys = state_dicts[0].keys()

    for state_dict in state_dicts[1:]:
        if state_dict.keys() != reference_keys:
            raise ValueError("Client update keys do not match")

    averaged = {}

    for name, reference in state_dicts[0].items():
        if not torch.is_floating_point(reference):
            raise ValueError(
                f"SCAFFOLD delta {name} must be floating point"
            )

        result = torch.zeros_like(reference)

        # averaging over clients
        for state_dict in state_dicts:
            tensor = state_dict[name]

            if tensor.shape != reference.shape:
                raise ValueError(f"Shape mismatch for {name}")

            result.add_(tensor, alpha=1.0 / len(state_dicts))

        averaged[name] = result

    return averaged
