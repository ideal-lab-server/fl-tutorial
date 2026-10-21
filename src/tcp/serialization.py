import json
import struct

import numpy as np
import torch


def serialize_state_dict(state_dict):
    """Serialize a PyTorch state_dict into metadata and binary payload."""

    metadata = []
    payload_parts = []
    offset = 0

    for name, tensor in state_dict.items():
        array = tensor.detach().cpu().contiguous().numpy()
        data = array.tobytes()

        metadata.append({
            "name": name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "offset": offset,
            "nbytes": len(data),
        })

        payload_parts.append(data)
        offset += len(data)

    metadata_bytes = json.dumps(metadata).encode("utf-8")
    payload = b"".join(payload_parts)

    # Prefix contains the length of the JSON metadata.
    prefix = struct.pack("!I", len(metadata_bytes))

    return prefix + metadata_bytes + payload


def deserialize_state_dict(data):
    """Reconstruct a PyTorch state_dict from serialized bytes."""

    prefix_size = struct.calcsize("!I")
    metadata_size = struct.unpack("!I", data[:prefix_size])[0]

    metadata_start = prefix_size
    metadata_end = metadata_start + metadata_size

    metadata = json.loads(data[metadata_start:metadata_end].decode("utf-8"))
    payload = data[metadata_end:]

    state_dict = {}

    for item in metadata:
        start = item["offset"]
        end = start + item["nbytes"]

        array = np.frombuffer(
            payload[start:end],
            dtype=np.dtype(item["dtype"]),
        ).copy()

        array = array.reshape(item["shape"])
        state_dict[item["name"]] = torch.from_numpy(array)

    return state_dict