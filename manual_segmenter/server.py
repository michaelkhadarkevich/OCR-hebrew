#!/usr/bin/env python3
import base64
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent


def normalize_output_path(raw):
    if not raw:
        raise ValueError("empty output folder")
    path = raw.strip().strip('"')
    path = path.replace("\\", "/")
    lower = path.lower()
    if lower.startswith("//wsl.localhost/ubuntu"):
        path = path[len("//wsl.localhost/Ubuntu"):]
        if not path.startswith("/"):
            path = "/" + path
    elif len(path) >= 3 and path[1] == ":" and path[2] == "/":
        drive = path[0].lower()
        path = f"/mnt/{drive}/{path[3:]}"
    return Path(path).expanduser().resolve()


def safe_name(name):
    name = unquote(str(name)).replace("\\", "/").split("/")[-1]
    if not name or name in {".", ".."}:
        raise ValueError("bad file name")
    return name


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/save_page":
            self.send_json(404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            out_dir = normalize_output_path(payload.get("output_folder", ""))
            out_dir.mkdir(parents=True, exist_ok=True)
            saved = []
            for item in payload.get("files", []):
                name = safe_name(item.get("name", ""))
                target = out_dir / name
                if item.get("kind") == "text":
                    target.write_text(item.get("text", ""), encoding="utf-8")
                else:
                    data_url = item.get("data_url", "")
                    if "," in data_url:
                        data_url = data_url.split(",", 1)[1]
                    target.write_bytes(base64.b64decode(data_url))
                saved.append(str(target))
            self.send_json(200, {"ok": True, "output_folder": str(out_dir), "saved": saved})
        except Exception as exc:
            self.send_json(500, {"ok": False, "error": str(exc)})


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8765
    print(f"Manual segmenter server: http://localhost:{port}")
    print(f"Serving: {ROOT}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()