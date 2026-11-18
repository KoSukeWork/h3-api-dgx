import os
import socket

import pytest

from h3_api.backend import check_backend_port_available


def test_port_probe_rejects_live_listener():
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen()
        with pytest.raises(OSError):
            check_backend_port_available(server.getsockname()[1])


@pytest.mark.skipif(os.name != "posix", reason="Linux deployment TCP TIME_WAIT semantics")
def test_port_probe_allows_closed_listener_with_time_wait():
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
        server.listen()
        with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
            accepted, _ = server.accept()
            accepted.close()  # Server active-close: its accepted socket enters TIME_WAIT.
            assert client.recv(1) == b""
    check_backend_port_available(port)
