"""A tiny mock LSP server for tests: Content-Length JSON-RPC over stdio.

Modes (env ``MOCK_LSP_MODE``):
- ``echo``   — jsonrpc client tests: echo/fail/die requests, poke/ask-client
- ``pull``   — (default) textDocument/diagnostic returns canned diagnostics
- ``push``   — didOpen/didChange triggers publishDiagnostics; pull errors
- ``crash``  — exits mid-request on textDocument/diagnostic
- ``hang``   — never answers ``initialize``

Canned diagnostics come from env ``MOCK_LSP_DIAGS`` (JSON list, LSP shape).
When ``MOCK_LSP_LOG`` is set, every received method name is appended there.
"""

from __future__ import annotations

import json
import os
import sys

MODE = os.environ.get("MOCK_LSP_MODE", "pull")
LOG = os.environ.get("MOCK_LSP_LOG")

DEFAULT_DIAGS = [
    {
        "range": {"start": {"line": 0, "character": 4}, "end": {"line": 0, "character": 7}},
        "severity": 1,
        "message": "undefined name 'foo'",
        "source": "mock",
    },
    {
        "range": {"start": {"line": 2, "character": 0}, "end": {"line": 2, "character": 3}},
        "severity": 2,
        "message": "unused variable",
        "source": "mock",
    },
]

DIAGS = json.loads(os.environ.get("MOCK_LSP_DIAGS") or json.dumps(DEFAULT_DIAGS))


def log(method: str) -> None:
    if LOG:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(method + "\n")


def send(payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
    sys.stdout.buffer.flush()


def respond(request_id, result=None, error=None) -> None:
    payload = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        payload["error"] = error
    else:
        payload["result"] = result
    send(payload)


def read_exact(n: int) -> bytes | None:
    chunks = b""
    while len(chunks) < n:
        chunk = sys.stdin.buffer.read(n - len(chunks))
        if not chunk:
            return None
        chunks += chunk
    return chunks


def read_message() -> dict | None:
    length = None
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            break
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1].strip())
    if length is None:
        return None
    body = read_exact(length)
    return json.loads(body.decode("utf-8")) if body is not None else None


def handle_request(request_id, method: str, params: dict) -> None:
    if method == "initialize":
        if MODE == "hang":
            return  # never answered — the client must time out
        respond(request_id, {"capabilities": {"diagnosticProvider": {}}})
    elif method == "shutdown":
        respond(request_id, None)
    elif method == "echo":
        respond(request_id, params)
    elif method == "fail":
        respond(request_id, error={"code": -32601, "message": "method not found"})
    elif method == "die":
        os._exit(1)
    elif method == "textDocument/diagnostic":
        if MODE == "crash":
            os._exit(1)
        if MODE == "push":
            respond(request_id, error={"code": -32601, "message": "pull unsupported"})
        else:
            respond(request_id, {"kind": "full", "items": DIAGS})
    else:
        respond(request_id, error={"code": -32601, "message": f"unknown: {method}"})


def handle_notification(method: str, params: dict) -> None:
    if method == "exit":
        sys.exit(0)
    if method == "poke":
        send({"jsonrpc": "2.0", "method": "poked", "params": params})
    elif method == "ask-client":
        send({"jsonrpc": "2.0", "id": 999, "method": "workspace/configuration", "params": params})
    elif method in ("textDocument/didOpen", "textDocument/didChange") and MODE == "push":
        uri = (params.get("textDocument") or {}).get("uri", "")
        send(
            {
                "jsonrpc": "2.0",
                "method": "textDocument/publishDiagnostics",
                "params": {"uri": uri, "diagnostics": DIAGS},
            }
        )


def main() -> None:
    while True:
        message = read_message()
        if message is None:
            return
        if "method" not in message:
            if message.get("id") == 999:  # reply to our server→client request
                send(
                    {
                        "jsonrpc": "2.0",
                        "method": "got-reply",
                        "params": {"value": message.get("result")},
                    }
                )
            continue
        method = str(message["method"])
        log(method)
        if "id" in message:
            handle_request(message["id"], method, message.get("params") or {})
        else:
            handle_notification(method, message.get("params") or {})


if __name__ == "__main__":
    main()
