#!/usr/bin/env python3
import base64
import json
import os
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

ROOT = Path(__file__).resolve().parent
PID_FILE = ROOT / "manual_segmenter.pid"


def normalize_output_path(raw):
    if not raw:
        raise ValueError("empty output folder")
    path = raw.strip().strip('"').replace("\\", "/")
    lower = path.lower()
    if os.name == "nt":
        return Path(path).expanduser().resolve()
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


def decode_data_url(data_url):
    if not isinstance(data_url, str) or not data_url:
        raise ValueError("missing file data")
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return base64.b64decode(data_url)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        # Always serve the UI uncached so an update cannot leave an old HTML/JS copy in the browser.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path == "/save_page":
            self.handle_save_page()
        elif self.path == "/render_pdf":
            self.handle_render_pdf()
        else:
            self.send_json(404, {"ok": False, "error": "not found"})

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("empty request")
        # Large PDFs are allowed, but reject unreasonable accidental requests.
        if length > 300 * 1024 * 1024:
            raise ValueError("request is too large")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def handle_save_page(self):
        try:
            payload = self.read_json()
            out_dir = normalize_output_path(payload.get("output_folder", ""))
            out_dir.mkdir(parents=True, exist_ok=True)
            saved = []
            for item in payload.get("files", []):
                name = safe_name(item.get("name", ""))
                target = out_dir / name
                if item.get("kind") == "text":
                    target.write_text(item.get("text", ""), encoding="utf-8")
                else:
                    target.write_bytes(decode_data_url(item.get("data_url", "")))
                saved.append(str(target))
            self.send_json(200, {"ok": True, "output_folder": str(out_dir), "saved": saved})
        except Exception as exc:
            self.send_json(500, {"ok": False, "error": str(exc)})

    def handle_render_pdf(self):
        try:
            if fitz is None:
                raise RuntimeError("PDF support is not installed. Re-run the installer.")
            payload = self.read_json()
            original_name = safe_name(payload.get("name", "document.pdf"))
            pdf_bytes = decode_data_url(payload.get("data_url", ""))
            scale = float(payload.get("scale", 2.0))
            scale = max(1.0, min(3.0, scale))
            stem = Path(original_name).stem

            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            try:
                if doc.page_count > 250:
                    raise ValueError("PDF has more than 250 pages; split it into smaller files first")
                pages = []
                matrix = fitz.Matrix(scale, scale)
                for index in range(doc.page_count):
                    page = doc.load_page(index)
                    pix = page.get_pixmap(matrix=matrix, alpha=False)
                    png = pix.tobytes("png")
                    encoded = base64.b64encode(png).decode("ascii")
                    pages.append({
                        "name": f"{stem}_p{index + 1:04d}.png",
                        "data_url": "data:image/png;base64," + encoded,
                    })
            finally:
                doc.close()
            self.send_json(200, {"ok": True, "pages": pages})
        except Exception as exc:
            self.send_json(500, {"ok": False, "error": str(exc)})


def open_app(url):
    try:
        webbrowser.open(url)
    except Exception:
        pass


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8770
    url = f"http://localhost:{port}/?v=pdf-portable-v4"
    print("Manual Segmenter - PDF Portable v4")
    print(f"Opening: {url}")
    print("Keep this window open while using the app.")
    print("Close this window to stop the app.\n")
    print(f"Serving: {ROOT}")

    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except OSError as exc:
        print(f"Could not start server on port {port}: {exc}")
        print("If Manual Segmenter is already running, open http://localhost:8770")
        input("Press Enter to close...")
        raise SystemExit(1)

    try:
        PID_FILE.write_text(str(os.getpid()), encoding="ascii")
    except Exception:
        pass

    threading.Timer(0.8, open_app, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        try:
            if PID_FILE.exists() and PID_FILE.read_text(encoding="ascii").strip() == str(os.getpid()):
                PID_FILE.unlink()
        except Exception:
            pass
