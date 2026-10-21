import json
import socket
import struct
from .common import MAX_HEADER_SIZE, MAX_PAYLOAD_SIZE


PREFIX_FORMAT = "!II"
PREFIX_SIZE = struct.calcsize(PREFIX_FORMAT)


def recv_exact(sock: socket.socket, size: int) -> bytes:
    """Read exactly size bytes from a TCP socket."""
    
    chunks = []
    remaining = size

    while remaining > 0:
        chunk = sock.recv(remaining)      # receives up to remaining bytes via TCP

        if not chunk:
            raise ConnectionError("Peer closed the connection")

        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def send_message(
    sock: socket.socket, 
    header: dict, 
    payload: bytes = b"",
    ) -> None:
    """Send one length-prefixed application message."""
    
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")

    header_bytes = json.dumps(header).encode("utf-8")

    if not 0 < len(header_bytes) <= MAX_HEADER_SIZE:
        raise ValueError("Invalid header size")

    if len(payload) > MAX_PAYLOAD_SIZE:
        raise ValueError("Payload exceeds maximum size")

    prefix = struct.pack(
        PREFIX_FORMAT, 
        len(header_bytes), 
        len(payload),
        )

    sock.sendall(prefix + header_bytes + payload)


def recv_message(sock: socket.socket) -> tuple[dict, bytes]:
    """Receive exactly one application message."""
    
    prefix = recv_exact(sock, PREFIX_SIZE)
    header_size, payload_size = struct.unpack(PREFIX_FORMAT, prefix)

    if not 0 < header_size <= MAX_HEADER_SIZE:
        raise ValueError("Invalid header size")

    if payload_size > MAX_PAYLOAD_SIZE:
        raise ValueError("Payload exceeds maximum size")

    header_bytes = recv_exact(sock, header_size)
    payload = recv_exact(sock, payload_size)
    header = json.loads(header_bytes.decode("utf-8"))

    if not isinstance(header, dict):
        raise ValueError("Header must be a JSON object")

    return header, payload
