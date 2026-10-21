import torch


def aggregate_state_dicts(client_state_dicts, client_sizes):
    """
    Aggregate client model state dictionaries using sample-size weights.

    Args:
        client_state_dicts: List of local model state dictionaries.
        client_sizes: Number of training examples at each client.

    Returns:
        A new aggregated state dictionary.
    """

    if not client_state_dicts:
        raise ValueError("At least one client update is required")

    if len(client_state_dicts) != len(client_sizes):
        raise ValueError(
            "Number of client models and client sizes must match"
        )

    if any(size <= 0 for size in client_sizes):
        raise ValueError("Client sizes must be positive")

    total_samples = sum(client_sizes)
    weights = [size / total_samples for size in client_sizes]

    reference_keys = client_state_dicts[0].keys()

    for state_dict in client_state_dicts[1:]:
        if state_dict.keys() != reference_keys:
            raise ValueError("Client state_dict keys do not match")

    aggregated = {}

    for name in reference_keys:
        reference_tensor = client_state_dicts[0][name]

        if not torch.is_floating_point(reference_tensor):
            # Non-floating buffers (e.g., integer counters) cannot
            # meaningfully be averaged as model parameters.
            if any(
                not torch.equal(state_dict[name], reference_tensor)
                for state_dict in client_state_dicts[1:]
            ):
                raise ValueError(
                    f"Cannot average differing non-floating tensor: {name}"
                )

            aggregated[name] = reference_tensor.clone()
            continue

        result = torch.zeros_like(reference_tensor)

        for state_dict, weight in zip(client_state_dicts, weights):
            if state_dict[name].shape != reference_tensor.shape:
                raise ValueError(
                    f"Tensor shape mismatch for {name}"
                )

            result.add_(state_dict[name], alpha=weight)

        aggregated[name] = result

    return aggregated
