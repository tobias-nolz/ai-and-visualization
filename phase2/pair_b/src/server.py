"""Serve the Phase 2 Pair B single-screen white-box visualization on `/`.

Usage (from repo root, WSL or Windows):
    python phase2/pair_b/src/server.py [--port 8875]

Routes:
    http://127.0.0.1:8875/                                 Inside Camera 156 (Option B)
    http://127.0.0.1:8875/results/camera156_whitebox.json  Precomputed white-box bundle
    http://127.0.0.1:8875/images/<filename>[?mode=model&mask_bar=1]
"""

from __future__ import annotations

import argparse
import io
import mimetypes
import sys
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image

APP = Path(__file__).resolve().parents[1]
REPO = APP.parents[1]
WEB = APP / "src" / "web"
RESULTS = APP / "results"
DEFAULT_DATA_DIR = REPO / "data" / "iwildcam_v2.0" / "train"
WSL_FALLBACK_DIR = Path(r"\\wsl.localhost\Ubuntu\home\martin\ai-and-visualization\data\iwildcam_v2.0\train")


def resolve_image_dir(requested: Path) -> Path:
    if requested.is_dir():
        return requested
    if sys.platform == "win32" and requested == DEFAULT_DATA_DIR and WSL_FALLBACK_DIR.is_dir():
        return WSL_FALLBACK_DIR
    return requested


@lru_cache(maxsize=512)
def render_model_jpeg(image_path_str: str, mask_bar: bool) -> bytes:
    with Image.open(image_path_str) as raw:
        img = raw.convert("RGB").resize((448, 448), Image.Resampling.BILINEAR)
    if mask_bar:
        # Black out exact bottom 24 px bar (rows 424..447) in 448x448 model space
        black = Image.new("RGB", (448, 24), (0, 0, 0))
        img.paste(black, (0, 424))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, request, address, server):
        super().__init__(request, address, server, directory=str(WEB))

    def send_bytes(self, body: bytes, content_type: str, cache_control: str = "no-store") -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        if not any(k.lower() == "cache-control" for k, _ in getattr(self, "_headers_buffer_seen", [])):
            pass
        super().end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            target = WEB / "index.html"
            if target.is_file():
                self.send_bytes(target.read_bytes(), "text/html; charset=utf-8", "no-store")
            else:
                self.send_error(404, "index.html not found")
            return

        if path.startswith("/results/") and path.endswith(".json"):
            fname = Path(unquote(path.removeprefix("/results/"))).name
            f_path = RESULTS / fname
            if f_path.is_file():
                self.send_bytes(f_path.read_bytes(), "application/json; charset=utf-8", "no-store")
            else:
                self.send_error(404, f"Result {fname} not found")
            return

        if path.startswith("/images/"):
            raw_id = Path(unquote(path.removeprefix("/images/"))).name
            if not raw_id.endswith(".jpg"):
                raw_id = f"{raw_id}.jpg"
            img_path = self.server.image_dir / raw_id
            if not img_path.is_file():
                self.send_error(404, f"Image {raw_id} not found")
                return
            mode = qs.get("mode", ["native"])[0]
            mask_bar = qs.get("mask_bar", ["0"])[0] == "1"
            if mode == "model" or mask_bar:
                body = render_model_jpeg(str(img_path), mask_bar)
                self.send_bytes(body, "image/jpeg", "public, max-age=3600")
            else:
                ctype = mimetypes.guess_type(img_path.name)[0] or "image/jpeg"
                self.send_bytes(img_path.read_bytes(), ctype, "public, max-age=3600")
            return

        # Static assets (e.g. app.js) served with no-store
        target = WEB / path.lstrip("/")
        if target.is_file() and target.resolve().parent == WEB.resolve():
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self.send_bytes(target.read_bytes(), ctype, "no-store")
            return

        self.send_error(404, f"Not found: {path}")

    def log_message(self, format: str, *args) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8875, help="Port to listen on (default: 8875)")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory with iWildCam train/*.jpg")
    args = parser.parse_args()

    WEB.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    img_dir = resolve_image_dir(args.images_dir)
    if not img_dir.is_dir():
        print(
            f"[WARNING] Image directory not found: {img_dir}\n"
            f"          Pass --images-dir <path-to-iwildcam_v2.0/train> or run inside WSL.",
            file=sys.stderr,
            flush=True,
        )

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.image_dir = img_dir
    print(f"Phase 2 Pair B Server: http://127.0.0.1:{args.port}/ (images: {img_dir})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
