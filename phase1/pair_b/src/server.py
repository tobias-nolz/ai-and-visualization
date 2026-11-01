"""Serve the atlas and image-review tool locally."""
import csv
import json
import mimetypes
import threading
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

APP = Path(__file__).resolve().parents[1]
WEB = APP / 'src/web'
LOCK = threading.Lock()


def load_images(manifest):
    with manifest.open(newline='', encoding='utf-8-sig') as file:
        rows = list(csv.DictReader(file))
    required = {'image_id', 'seq_id', 'camera', 'datetime', 'label', 'image_path'}
    if not rows or not required <= rows[0].keys():
        raise ValueError(f'Manifest needs columns: {sorted(required)}')
    if len({row['image_id'] for row in rows}) != len(rows):
        raise ValueError('Image IDs must be unique.')
    paths = {}
    sequences = {}
    for row in sorted(rows, key=lambda r: (r['camera'], r['datetime'], r['image_id'])):
        image_path = Path(row['image_path']).expanduser()
        if not image_path.is_absolute():
            image_path = manifest.parent / image_path
        paths[row['image_id']] = image_path
        sequence = sequences.setdefault(row['seq_id'], {
            'id': row['seq_id'], 'camera': row['camera'], 'frames': []
        })
        sequence['frames'].append({
            'id': row['image_id'], 'label': row['label'], 'time': row['datetime'],
            'url': '/images/' + row['image_id']
        })
    for sequence in sequences.values():
        for index, frame in enumerate(sequence['frames'], 1):
            frame['index'] = index
    data = {'imageCount': len(rows), 'sequences': list(sequences.values())}
    return data, paths


def merge_state(existing, incoming, image_labels):
    annotations = existing.setdefault('annotations', {})
    for image_id, annotation in incoming.get('annotations', {}).items():
        if image_id not in image_labels:
            continue
        if annotation['status'] not in ['phantom_label', 'not_flagged']:
            raise ValueError('Invalid review status.')
        if 'absent' in annotation:
            if not isinstance(annotation['absent'], bool):
                raise ValueError('Absent must be true or false.')
            animal_label = image_labels[image_id] not in ['empty', 'motorcycle']
            annotation['status'] = 'phantom_label' if annotation['absent'] and animal_label else 'not_flagged'
        if annotation.get('changed_utc', '') >= annotations.get(image_id, {}).get('changed_utc', ''):
            annotations[image_id] = annotation
    existing['cursor'] = incoming.get('cursor', {})
    existing['updated_utc'] = datetime.now(timezone.utc).isoformat()
    return existing


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, request, address, server):
        super().__init__(request, address, server, directory=str(WEB))

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def send_bytes(self, body, content_type):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, value):
        self.send_bytes(json.dumps(value).encode(), 'application/json')

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/':
            self.send_bytes((WEB / 'atlas.html').read_bytes(), 'text/html; charset=utf-8')
        elif path in ['/viewer/', '/viewer']:
            self.send_bytes((WEB / 'review.html').read_bytes(), 'text/html; charset=utf-8')
        elif path == '/results/atlas_data.json':
            self.send_bytes((APP / 'results/atlas_data.json').read_bytes(), 'application/json')
        elif path == '/api/species-names':
            with (APP.parents[1] / 'data/species_groups.csv').open(newline='', encoding='utf-8') as file:
                names = {row['name']: row['common_name'] for row in csv.DictReader(file)}
            self.send_json(names)
        elif path == '/api/images':
            self.send_json(self.server.images)
        elif path == '/api/manual-review':
            with LOCK:
                self.send_json(self.server.read_state())
        elif path.startswith('/images/'):
            image_id = unquote(path.removeprefix('/images/'))
            image = self.server.image_paths.get(image_id)
            if image is None:
                image = self.server.image_dir / (Path(image_id).name + '.jpg')
            if image and image.is_file():
                self.send_bytes(image.read_bytes(), mimetypes.guess_type(image.name)[0] or 'image/jpeg')
            else:
                self.send_error(404)
        else:
            super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != '/api/manual-review':
            self.send_error(404)
            return
        origin = self.headers.get('Origin')
        allowed_origins = [f'http://{host}:{self.server.server_port}' for host in ['127.0.0.1', 'localhost']]
        if origin and origin not in allowed_origins:
            self.send_error(403)
            return
        if self.headers.get_content_type() != 'application/json':
            self.send_error(415)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 3000000:
                self.send_error(413)
                return
            incoming = json.loads(self.rfile.read(length))
            with LOCK:
                state = merge_state(self.server.read_state(), incoming, self.server.image_labels)
                self.server.save_state(state)
            self.send_json({'saved': True})
        except (ValueError, TypeError, KeyError):
            self.send_error(400)

    def log_message(self, format, *args):
        pass


def main():
    results = APP / 'results'
    output = results / 'human_review.json'
    images, paths = load_images(results / 'viewer_images.csv')
    results.mkdir(exist_ok=True)

    def read_state():
        if not output.exists():
            return {'annotations': {}, 'cursor': {}}
        return json.loads(output.read_text(encoding='utf-8'))

    def save_state(state):
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temporary.replace(output)

    server = ThreadingHTTPServer(('127.0.0.1', 8874), Handler)
    server.images, server.image_paths = images, paths
    server.image_labels = {frame['id']: frame['label']
                           for sequence in images['sequences'] for frame in sequence['frames']}
    server.image_dir = next(iter(paths.values())).parent
    server.read_state, server.save_state = read_state, save_state
    print('Atlas: http://127.0.0.1:8874/', flush=True)
    print(f'Viewer: http://127.0.0.1:8874/viewer/ ({len(paths)} images)', flush=True)
    print(f'Autosave: {output}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
