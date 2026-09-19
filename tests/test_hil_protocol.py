import numpy as np

from scripts.hil_protocol import REQUEST_HEADER, RESPONSE_HEADER, pack_request, unpack_response


def test_request_packet_contains_fixed_header_and_two_observation_vectors():
    encoder = np.arange(1247, dtype=np.float32)
    decoder = np.arange(994, dtype=np.float32) + 1

    packet = pack_request(7, encoder, decoder)

    assert len(packet) == REQUEST_HEADER.size + (1247 + 994) * 4
    assert packet[:4] == b"SONC"
    assert int.from_bytes(packet[8:12], "little") == 7


def test_response_packet_decodes_sequence_and_action():
    action = np.arange(29, dtype=np.float32)
    header = RESPONSE_HEADER.pack(b"SONC", 1, 2, 11, 0)

    sequence, status, decoded = unpack_response(header + action.tobytes())

    assert sequence == 11
    assert status == 0
    np.testing.assert_array_equal(decoded, action)
