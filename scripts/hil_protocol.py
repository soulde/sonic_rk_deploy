"""Fixed-size TCP protocol shared by the simulator and RK3576 server."""

from __future__ import annotations

import struct

import numpy as np

MAGIC = b"SONC"
VERSION = 1
REQUEST_TYPE = 1
RESPONSE_TYPE = 2
STATUS_OK = 0
STATUS_BAD_PACKET = 1
REQUEST_HEADER = struct.Struct("<4sHHII")
RESPONSE_HEADER = struct.Struct("<4sHHII")
ENCODER_ELEMENTS = 1247
DECODER_ELEMENTS = 994
ACTION_ELEMENTS = 29


def pack_request(sequence: int, encoder: np.ndarray, decoder: np.ndarray) -> bytes:
    encoder = np.asarray(encoder, dtype=np.float32).reshape(-1)
    decoder = np.asarray(decoder, dtype=np.float32).reshape(-1)
    if encoder.size != ENCODER_ELEMENTS or decoder.size != DECODER_ELEMENTS:
        raise ValueError("request must contain [1247] encoder and [994] decoder elements")
    header = REQUEST_HEADER.pack(MAGIC, VERSION, REQUEST_TYPE, int(sequence), 0)
    return header + np.concatenate((encoder, decoder)).astype("<f4", copy=False).tobytes()


def unpack_response(packet: bytes) -> tuple[int, int, np.ndarray]:
    expected_size = RESPONSE_HEADER.size + ACTION_ELEMENTS * 4
    if len(packet) != expected_size:
        raise ValueError(f"response size {len(packet)} != {expected_size}")
    magic, version, message_type, sequence, status = RESPONSE_HEADER.unpack_from(packet)
    if magic != MAGIC or version != VERSION or message_type != RESPONSE_TYPE:
        raise ValueError("invalid response header")
    action = np.frombuffer(packet, dtype="<f4", offset=RESPONSE_HEADER.size, count=ACTION_ELEMENTS).copy()
    return sequence, status, action
