#!/usr/bin/env python3
"""Check and pack a team's Round-1 submission (AI Drone Challenge FA26).

Produces submission_<team_id>.tar.gz plus submission_<team_id>.sha256 (sha256sum format).
Checks: entry script exists and parses as Python 3.10 (the grading image), every other .py
parses, and the package (code + .pt/.onnx weights) is at most 30 MiB.

    python3 pack_submission.py --team-id T07
    python3 pack_submission.py --team-id T07 --entry solution.py --args "--alt 12 --speed 3"
    python3 pack_submission.py --team-id T07 --upload http://<server>:8765
    python3 pack_submission.py --team-id T07 --select 2

Standard library only; runs on the laptop (Windows/WSL) or inside the kit container.
"""
import argparse
import ast
import datetime
import fnmatch
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import urllib.request

MAX_PACKAGE_BYTES = 30 * 1024 * 1024  # 30 MiB, also enforced by the server receiver
TEAM_ID_RE = re.compile(r'^[A-Za-z0-9_-]{1,32}$')
ENTRY_CANDIDATES = ('solution.py', 'fly_survey_baylands.py')
WEIGHT_SUFFIXES = ('.pt', '.onnx')
# Allowed manifest env overrides; the runner ignores anything else.
ALLOWED_ENV = ('KIT_CAM_PITCH_DEG', 'KIT_CAM_RES', 'KIT_YOLO', 'YOLO_WEIGHTS', 'KIT_DRAW_BBOX')
# Kit-provided or generated content that never belongs in a submission.
EXCLUDE_DIRS = {'logs', 'flight-results', 'output', 'dist', 'drone_cv', '__pycache__', '.git',
                '.venv', 'venv', '.grading', '.ipynb_checkpoints', '.pytest_cache'}
TOKEN_FILE = '.drone_team_token'
EXCLUDE_FILES = ['submission_*.tar.gz', 'submission_*.sha256', '*.pyc', '*.log', '.DS_Store', TOKEN_FILE]
GRADING_PYTHON = (3, 10)


class PackError(Exception):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def collect_files(src, extra_excludes=()):
    """Return sorted relative POSIX paths to include; symlinks are refused."""
    patterns = EXCLUDE_FILES + list(extra_excludes)
    files = []
    for root, dirs, names in os.walk(src):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.'))
        for name in sorted(names):
            path = Path(root, name)
            rel = path.relative_to(src).as_posix()
            if any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(rel, p) for p in patterns):
                continue
            if path.is_symlink():
                raise PackError(f'symlink not allowed: {rel}')
            if path.is_file():
                files.append(rel)
    return files


def check_syntax(path, rel):
    try:
        ast.parse(Path(path).read_bytes(), filename=rel, feature_version=GRADING_PYTHON)
    except SyntaxError as exc:
        raise PackError(f'syntax error in {rel}:{exc.lineno}: {exc.msg} '
                        f'(grading image runs Python {GRADING_PYTHON[0]}.{GRADING_PYTHON[1]})') from exc


def pick_entry(src, files, entry):
    if entry:
        if entry not in files:
            raise PackError(f'entry script {entry!r} not found in {src}')
        return entry
    for name in ENTRY_CANDIDATES:
        if name in files:
            return name
    raise PackError(f'no entry script: need one of {", ".join(ENTRY_CANDIDATES)} (or --entry)')


def parse_env(items):
    env = {}
    for item in items:
        key, sep, value = item.partition('=')
        if not sep or key not in ALLOWED_ENV:
            raise PackError(f'--env {item!r}: expected KEY=VALUE with KEY in {", ".join(ALLOWED_ENV)}')
        env[key] = value
    return env


def build(src, team_id, out_dir, entry=None, args='', env=None, excludes=()):
    """Validate and write the archive. Returns (archive_path, sha_path, manifest)."""
    if not TEAM_ID_RE.match(team_id):
        raise PackError('team id must match [A-Za-z0-9_-]{1,32}')
    src, out_dir = Path(src).resolve(), Path(out_dir).resolve()
    if not src.is_dir():
        raise PackError(f'source directory not found: {src}')
    files = collect_files(src, excludes)
    entry = pick_entry(src, files, entry)
    for rel in files:
        if rel.endswith('.py'):
            check_syntax(src / rel, rel)
    entries = [dict(path=rel, size=(src / rel).stat().st_size, sha256=sha256_file(src / rel)) for rel in files]
    total = sum(e['size'] for e in entries)
    weights = [e for e in entries if e['path'].lower().endswith(WEIGHT_SUFFIXES)]
    if total > MAX_PACKAGE_BYTES:
        biggest = sorted(entries, key=lambda e: -e['size'])[:5]
        detail = ', '.join(f"{e['path']} {e['size']/2**20:.1f} MiB" for e in biggest)
        raise PackError(f'package is {total/2**20:.2f} MiB > 30 MiB limit (largest: {detail})')
    manifest = dict(team_id=team_id, entry=entry, args=args, env=env or {},
                    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    total_bytes=total, weights=[e['path'] for e in weights], files=entries)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f'submission_{team_id}.tar.gz'
    tmp = archive.with_suffix('.tmp')
    with tarfile.open(tmp, 'w:gz') as tar:
        for rel in files:
            info = tar.gettarinfo(src / rel, arcname=rel)
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            info.mode = 0o644
            with open(src / rel, 'rb') as f:
                tar.addfile(info, f)
        data = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo('manifest.json')
        info.size, info.mode = len(data), 0o644
        info.mtime = int(datetime.datetime.now().timestamp())
        tar.addfile(info, io.BytesIO(data))
    if tmp.stat().st_size > MAX_PACKAGE_BYTES:
        tmp.unlink()
        raise PackError('compressed archive exceeds 30 MiB')
    tmp.replace(archive)
    digest = sha256_file(archive)
    sha_path = out_dir / f'submission_{team_id}.sha256'
    sha_path.write_text(f'{digest}  {archive.name}\n')
    return archive, sha_path, manifest


DEFAULT_HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 pack_submission/1.0'}


def token_path():
    return Path.cwd() / TOKEN_FILE


def load_team_token(team_id):
    try:
        data = json.loads(token_path().read_text())
    except (OSError, ValueError):
        return {}
    if data.get('team_id') and data.get('team_id') != team_id:
        return {}
    return data


def save_team_token(team_id, token, server_url=None):
    data = {'team_id': team_id, 'token': token}
    if server_url:
        data['server_url'] = server_url.rstrip('/')
    token_path().write_text(json.dumps(data, indent=2) + '\n')


def upload(url, archive, sha_path, team_id, admin_key=None):
    digest = sha_path.read_text().split()[0]
    headers = dict(DEFAULT_HEADERS, **{'X-Submission-SHA256': digest, 'Content-Type': 'application/gzip'})
    if admin_key:
        headers['X-Admin-Key'] = admin_key
    req = urllib.request.Request(f"{url.rstrip('/')}/submit/{team_id}", data=archive.read_bytes(), method='PUT',
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def fetch_history(url, team_id):
    req = urllib.request.Request(f"{url.rstrip('/')}/history/{team_id}", headers=DEFAULT_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def resolve_selection(url, team_id, selection):
    value = str(selection)
    if not value.isdigit():
        return value
    history = fetch_history(url, team_id)
    submissions = history.get('submissions', [])
    index = int(value)
    if 1 <= index <= len(submissions):
        return submissions[index - 1]['filename']
    if index == 0 and submissions:
        return submissions[0]['filename']
    return value


def select_version(url, team_id, selection, token=None, admin_key=None):
    selected = resolve_selection(url, team_id, selection)
    payload = json.dumps({'selection': selected}).encode()
    headers = dict(DEFAULT_HEADERS, **{'Content-Type': 'application/json'})
    if token:
        headers['X-Team-Token'] = token
    if admin_key:
        headers['X-Admin-Key'] = admin_key
    req = urllib.request.Request(f"{url.rstrip('/')}/select/{team_id}", data=payload, method='POST',
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--team-id', help='team code issued by the organizers, e.g. T07')
    ap.add_argument('--src', default=str(Path(__file__).resolve().parent), help='folder to pack (default: student-ws)')
    ap.add_argument('--entry', help=f'main script (default: first of {", ".join(ENTRY_CANDIDATES)})')
    ap.add_argument('--args', default='', help='command-line arguments for the entry script, one string')
    ap.add_argument('--env', action='append', default=[], metavar='KEY=VALUE',
                    help=f'kit settings for the grading run, allowed: {", ".join(ALLOWED_ENV)}')
    ap.add_argument('--no-bbox', action='store_true',
                    help='record video with OSD HUD only (no YOLO bounding boxes; sets KIT_DRAW_BBOX=0)')
    ap.add_argument('--exclude', action='append', default=[], metavar='GLOB', help='extra files to leave out')
    ap.add_argument('--out', default='.', help='output folder (default: current directory)')
    ap.add_argument('--upload', metavar='URL', help='also upload to the receiver, e.g. http://server:8765')
    ap.add_argument('--admin-key', help='organizer key to bypass upload rate limits')
    ap.add_argument('--team-token', help='team token from a previous accepted upload')
    ap.add_argument('--select', metavar='VERSION', help='select an archived version by index, file name, or hash')
    ap.add_argument('--cancel', metavar='URL', help='cancel queued or running evaluation')
    ap.add_argument('--team-key', help='team cancellation key, if configured by organizers')
    ap.add_argument('--status', metavar='URL', help='view team result without packing or uploading')
    ap.add_argument('--status-selection', metavar='URL', help='confirm selected version for this team')
    ap.add_argument('--list-selections', metavar='URL', help='list all selected versions')
    ap.add_argument('--lock-selection', metavar='URL', help='lock student selection immediately')
    ap.add_argument('--extend-selection', nargs=2, metavar=('MINUTES', 'URL'), help='extend and reopen selection')
    a = ap.parse_args(argv)
    if not (a.list_selections or a.lock_selection or a.extend_selection) and not a.team_id:
        ap.error('--team-id is required')
    if a.status_selection or a.list_selections or a.lock_selection or a.extend_selection:
        if (a.lock_selection or a.extend_selection) and not a.admin_key:
            ap.error('selection control requires --admin-key')
        if a.status_selection and not TEAM_ID_RE.fullmatch(a.team_id):
            ap.error('invalid team ID')
        try:
            headers = dict(DEFAULT_HEADERS)
            if a.admin_key:
                headers['X-Admin-Key'] = a.admin_key
            payload = None
            if a.lock_selection or a.extend_selection:
                url = a.lock_selection or a.extend_selection[1]
                body = {'action': 'lock'} if a.lock_selection else {'action': 'extend', 'minutes': float(a.extend_selection[0])}
                payload = json.dumps(body).encode()
                headers['Content-Type'] = 'application/json'
                path = '/competition/selection'
            else:
                url = a.status_selection or a.list_selections
                path = '/selections/' + a.team_id if a.status_selection else '/selections'
            req = urllib.request.Request(url.rstrip('/') + path, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read())
            state = result['competition_state']
            print('Cổng chọn:', 'MỞ' if state['selection_open'] else 'ĐÃ KHÓA', '| Hạn:', state.get('selection_deadline'))
            for row in result.get('selections', []):
                print(f"{row['team_id']}: {row.get('archive', 'chưa chọn')} | trạng thái: {row['status']} | chốt: {row['finalized']}")
                print(f"  Nộp lúc: {row.get('submitted_at')} | chọn lúc: {row.get('selected_at')} | nguồn: {row.get('source')}")
                print(f"  SHA256: {row.get('sha256')}")
            if a.status_selection and not result.get('selections'):
                print(f'{a.team_id}: chưa có bài nộp / chưa chọn')
            return 0
        except Exception as exc:
            body = exc.read().decode(errors='replace') if hasattr(exc, 'read') else ''
            print(f'[selection] FAIL: {exc} {body}', file=sys.stderr)
            return 2
    if a.select:
        if not TEAM_ID_RE.fullmatch(a.team_id):
            ap.error('invalid team ID')
        saved = load_team_token(a.team_id)
        server_url = a.upload or saved.get('server_url')
        token = a.team_token or saved.get('token')
        if not server_url:
            ap.error('--select needs --upload http://<server>:8765 once, or a saved .drone_team_token from upload')
        if not token and not a.admin_key:
            ap.error('--select needs --team-token, --admin-key, or saved .drone_team_token')
        try:
            result = select_version(server_url, a.team_id, a.select, token, a.admin_key)
            print('[select] server:', json.dumps(result, ensure_ascii=False))
            return 0
        except Exception as exc:
            body = exc.read().decode(errors='replace') if hasattr(exc, 'read') else ''
            print(f'[select] FAIL: {exc} {body}', file=sys.stderr)
            return 2
    if a.cancel:
        if not TEAM_ID_RE.fullmatch(a.team_id):
            ap.error('invalid team ID')
        try:
            headers = dict(DEFAULT_HEADERS, **{'Content-Type': 'application/json'})
            if a.admin_key:
                headers['X-Admin-Key'] = a.admin_key
            if a.team_key:
                headers['X-Team-Key'] = a.team_key
            request = urllib.request.Request(f"{a.cancel.rstrip('/')}/cancel/{a.team_id}",
                data=b'{}', method='POST', headers=headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read())
            print(f"[cancel] {a.team_id}: {result.get('status')} — {result.get('message', '')}")
            return 0
        except Exception as exc:
            body = exc.read().decode(errors='replace') if hasattr(exc, 'read') else ''
            print(f'[cancel] FAIL: {exc} {body}', file=sys.stderr)
            return 2
    if a.status:
        if not TEAM_ID_RE.fullmatch(a.team_id):
            ap.error('invalid team ID')
        try:
            req = urllib.request.Request(f"{a.status.rstrip('/')}/result/{a.team_id}", headers=DEFAULT_HEADERS)
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read())

            print(f"\n=== KẾT QUẢ {a.team_id} · {result.get('status')} ===")
            for label, key in [('Tổng / 100','total'), ('AI','ai'), ('Tọa độ','coord'), ('An toàn','safety'), ('TP','tp'), ('FP','fp'), ('Duplicates','duplicates'), ('Sim time (s)','sim_time'), ('Wall time (s)','wall_time')]:
                value = result.get(key)
                print(f'{label:16}: {value if value is not None else chr(8212)}')
            print('Video:', a.status.rstrip('/') + result.get('video_url', f'/video/{a.team_id}'))
            print('--- Console (20 dòng cuối) ---')
            print('\n'.join(result.get('console_tail', [])) or '(chưa có log)')
            return 0

        except Exception as exc:
            print(f'[status] FAIL: {exc}', file=sys.stderr)
            return 2
    try:
        env = parse_env(a.env)
        if a.no_bbox:
            env['KIT_DRAW_BBOX'] = '0'
        archive, sha_path, manifest = build(a.src, a.team_id, a.out, a.entry, a.args, env, a.exclude)
    except PackError as exc:
        print(f'[pack] FAIL: {exc}', file=sys.stderr)
        return 1
    print(f"[pack] entry: {manifest['entry']} {manifest['args']}".rstrip())
    print(f"[pack] {len(manifest['files'])} files, {manifest['total_bytes']/2**20:.2f} MiB of 30 MiB")
    for w in manifest['weights']:
        print(f'[pack]   weights: {w}')
    print(f'[pack] OK: {archive}\n[pack]     {sha_path}: {sha_path.read_text().split()[0]}')
    if a.upload:
        try:
            result = upload(a.upload, archive, sha_path, a.team_id, a.admin_key)
            if result.get('team_token'):
                save_team_token(a.team_id, result['team_token'], a.upload)
                print(f'[pack] team token saved to {token_path()}')
            print('[pack] server:', json.dumps(result, ensure_ascii=False))
        except Exception as exc:  # network errors and HTTP 4xx both end here
            body = exc.read().decode(errors='replace') if hasattr(exc, 'read') else ''
            print(f'[pack] upload FAIL: {exc} {body}', file=sys.stderr)
            return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
