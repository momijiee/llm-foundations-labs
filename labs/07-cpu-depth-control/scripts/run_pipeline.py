"""CPU pipeline used directly by the experiment Notebook and optional CLI."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time
import uuid

import numpy as np
from PIL import Image
import psutil

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_RUNTIME = ROOT.parents[1] / 'lesson7' / 'runtime'
RUNTIME = Path(os.environ.get('HW7_RUNTIME', PRIVATE_RUNTIME if PRIVATE_RUNTIME.exists()
                           else ROOT / '.runtime')).resolve()
PROMPT = 'photorealistic render, studio soft lighting, PBR material, preserve original 3d geometry, sharp'
NEGATIVE = 'deformed geometry, perspective changed'


def executable(kind):
    override = os.environ.get('BLENDER_EXE' if kind == 'blender' else 'SD_CLI')
    if override:
        path = Path(override).expanduser().resolve()
    else:
        pattern = 'blender/**/blender.exe' if kind == 'blender' else 'sdcpp/**/sd-cli.exe'
        matches = sorted(RUNTIME.glob(pattern))
        if len(matches) != 1:
            raise FileNotFoundError(f'Expected one {kind} executable in runtime; set BLENDER_EXE / SD_CLI.')
        path = matches[0]
    if not path.is_file():
        raise FileNotFoundError(path.name)
    return path


def public_text(text):
    # Console output in the Notebook must not expose the local user directory.
    for p in [RUNTIME, ROOT, ROOT.parents[1], Path.home()]:
        for value in {str(p), p.as_posix()}:
            text = text.replace(value, '<local>')
    return text


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run_logged(command, log_path, cwd=None):
    """Stream stdout/stderr into Notebook and retain the unmodified local log."""
    command = [str(x) for x in command]
    print(public_text(subprocess.list2cmdline(command)), flush=True)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    peak_rss = 0
    import threading
    stop = threading.Event()
    proc = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding='utf-8', errors='replace', bufsize=1)

    def monitor():
        nonlocal peak_rss
        while not stop.wait(0.25):
            try:
                parent = psutil.Process(proc.pid)
                peak_rss = max(peak_rss, sum(p.memory_info().rss
                    for p in [parent] + parent.children(recursive=True) if p.is_running()))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        with log_path.open('w', encoding='utf-8') as log:
            log.write(subprocess.list2cmdline(command) + '\n')
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(public_text(line), end='', flush=True)
        code = proc.wait()
        if code:
            raise RuntimeError(f'Process exited with {code}; see {log_path.name}')
    finally:
        if proc.poll() is None:
            try:
                parent = psutil.Process(proc.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
            except psutil.NoSuchProcess:
                pass
            proc.wait()
        stop.set()
        thread.join(timeout=2)
        proc.stdout.close()
    return {'wall_s': time.perf_counter() - started,
            'peak_process_tree_rss_mib': peak_rss / 1024**2}


def environment():
    cpu = platform.processor()
    if platform.system() == 'Windows':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    return dict(python=platform.python_version(), os=platform.system(),
                cpu=cpu, physical_cores=psutil.cpu_count(logical=False),
                logical_cores=psutil.cpu_count(), ram_gib=round(psutil.virtual_memory().total / 1024**3, 2),
                available_ram_gib=round(psutil.virtual_memory().available / 1024**3, 2))


def ensure_rgb(path, expected_res):
    with Image.open(path) as img:
        if img.size != (expected_res, expected_res):
            raise ValueError(f'Control image size {img.size} does not match {expected_res}.')
        rgb = img.convert('RGB')
    # The caller provides a run-local copy, never modify the original control image.
    rgb.save(path)
    pixels = np.asarray(rgb)
    if pixels.min() == pixels.max():
        raise ValueError('Control image is constant; check Mist settings.')


def prepare_controls(res=512, samples=64, threads=12):
    out = RUNTIME / 'output' / f'controls_{res}'
    out.mkdir(parents=True, exist_ok=True)
    blend = RUNTIME / 'assets' / 'demo_white_model.blend'
    cmd = [executable('blender'), '-b', '--python-exit-code', '1', '-P',
           ROOT / 'blender' / 'make_demo_scene.py', '--', '--output', blend]
    scene_timing = run_logged(cmd, out / 'scene.log', ROOT)
    cmd = [executable('blender'), '-b', blend, '--python-exit-code', '1', '-P',
           ROOT / 'blender' / 'export_control_maps.py', '--', '--outdir', out,
           '--name', 'demo', '--res', res, '--samples', samples, '--threads', threads]
    export_timing = run_logged(cmd, out / 'export.log', ROOT)
    record = json.loads((out / 'demo_controls.json').read_text(encoding='utf-8'))
    record.update(scene_process=scene_timing, export_process=export_timing)
    record['script_sha256'] = {name: sha256(ROOT / 'blender' / name)
                              for name in ['make_demo_scene.py', 'export_control_maps.py']}
    record['blend_sha256'] = sha256(blend)
    record['files'] = {}
    for suffix in ['clay', 'depth_0001', 'normal_0001']:
        p = out / f'demo_{suffix}.png'
        with Image.open(p) as img:
            a = np.asarray(img)
            if img.size != (res, res) or a.min() == a.max():
                raise ValueError(f'Invalid image: {p.name}')
            record['files'][p.name] = dict(sha256=sha256(p), mode=img.mode,
                                          min=int(a.min()), max=int(a.max()))
    (out / 'preparation.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record


def read_controls(res=512):
    out = RUNTIME / 'output' / f'controls_{res}'
    p = out / 'preparation.json'
    if not p.is_file():
        raise FileNotFoundError('Controls not prepared; run prepare_controls() first.')
    record = json.loads(p.read_text(encoding='utf-8'))
    for name, digest in record['script_sha256'].items():
        if sha256(ROOT / 'blender' / name) != digest:
            raise ValueError('Preparation script changed; rerun prepare_controls().')
    if sha256(RUNTIME / 'assets' / 'demo_white_model.blend') != record['blend_sha256']:
        raise ValueError('Scene changed; rerun prepare_controls().')
    for name, info in record['files'].items():
        if sha256(out / name) != info['sha256']:
            raise ValueError('Control image changed; rerun preparation.')
    return record


def parse_timings(text):
    # Values absent from a backend version remain None, never inferred as measured.
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    patterns = {
        'sampling_s': [r'sampling completed, taking\s+([\d.]+)\s*s',
                       r'sampling completed[^(\n]*\(\s*([\d.]+)\s*s\s*\)',
                       r'sampling took\s+([\d.]+)\s*s'],
        'vae_decode_s': [r'decode_first_stage completed, taking\s+([\d.]+)\s*s',
                         r'decode_first_stage completed[^(\n]*\(\s*([\d.]+)\s*s\s*\)',
                         r'decoding completed[^(\n]*\(\s*([\d.]+)\s*s\s*\)',
                         r'vae decoding took\s+([\d.]+)\s*s'],
    }
    result = {}
    for name, options in patterns.items():
        matches = []
        for pattern in options:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if matches:
                break
        result[name] = float(matches[-1]) if matches else None
    return result


def run_experiment(label, steps=4, control_strength=0.9, res=512, cfg=1.2,
                   seed=42, threads=12, session='manual', control_image=None):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', label + session):
        raise ValueError('Use letters, digits, underscores or hyphens for label/session.')
    if res not in (512, 768) or steps < 1 or threads < 1:
        raise ValueError('Invalid resolution, steps or threads.')
    read_controls(res) if control_image is None else None
    models = RUNTIME / 'models'
    main_model = models / 'sd_turbo-q4_0.gguf'
    control_model = models / 'control_depth_sd21-fp16.gguf'
    for model in [main_model, control_model]:
        if not model.is_file() or model.stat().st_size < 100_000_000:
            raise FileNotFoundError(f'Missing/incomplete model: {model.name}')
        with model.open('rb') as f:
            if f.read(4) != b'GGUF':
                raise ValueError(f'Not GGUF: {model.name}')
    manifest = json.loads((RUNTIME / 'resources.json').read_text(encoding='utf-8'))
    for model in [main_model, control_model]:
        if sha256(model) != manifest['models'][model.name]['sha256']:
            raise ValueError(f'Model checksum changed: {model.name}')
    attempt = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    out = RUNTIME / 'output' / session / f'{label}-{attempt}'
    out.mkdir(parents=True, exist_ok=False)
    source = Path(control_image).resolve() if control_image else (
        RUNTIME / 'output' / f'controls_{res}' / 'demo_depth_0001.png')
    depth = out / 'depth.png'
    depth.write_bytes(source.read_bytes())
    ensure_rgb(depth, res)
    photo = out / f'{label}_photoreal.png'
    cli = executable('sdcpp')
    command = [cli, '-m', main_model, '-p', PROMPT, '-n', NEGATIVE,
               '--control-net', control_model, '--control-image', depth,
               '--control-strength', control_strength, '-W', res, '-H', res,
               '--steps', steps, '--cfg-scale', cfg, '--sampling-method', 'euler_a',
               '--scheduler', 'discrete', '--rng', 'cpu', '-s', seed, '-t', threads,
               '--backend', 'cpu', '--params-backend', 'cpu', '-o', photo]
    record = dict(label=label, session=session, attempt=attempt,
                  started_at=datetime.now().astimezone().isoformat(), status='running',
                  resolution=res, steps=steps, cfg=cfg, control_strength=control_strength,
                  seed=seed, threads=threads, sampler='euler_a', scheduler='discrete',
                  rng='cpu', backend='cpu', prompt=PROMPT, negative=NEGATIVE,
                  sdcpp_commit=manifest['sdcpp_commit'], models=manifest['models'],
                  depth_sha256=sha256(depth), environment=environment(),
                  photo=photo.name, log='sd-cli.log',
                  total_timing_scope='sd-cli process wall time; shared Blender preparation excluded')
    dest = out / 'record.json'
    dest.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    try:
        timing = run_logged(command, out / 'sd-cli.log', cli.parent)
        if not photo.is_file():
            raise RuntimeError('Backend exited without the requested output image.')
        with Image.open(photo) as img:
            if img.size != (res, res):
                raise RuntimeError('Generated image dimensions differ from requested dimensions.')
        record.update(status='complete', total_s=timing['wall_s'],
                      peak_process_tree_rss_mib=timing['peak_process_tree_rss_mib'],
                      photo_sha256=sha256(photo),
                      **parse_timings((out / 'sd-cli.log').read_text(encoding='utf-8')))
        if record['sampling_s'] is None or record['vae_decode_s'] is None:
            print('阶段耗时未完全识别；保留空值，请按原始日志核对。')
    except BaseException as exc:
        record.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                      error=public_text(str(exc)))
        raise
    finally:
        dest.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    # The absolute output path is returned for display but not stored in public records.
    return record, photo


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--name', default='test512')
    p.add_argument('--res', type=int, default=512, choices=[512, 768])
    p.add_argument('--steps', type=int, default=4)
    p.add_argument('--cfg', type=float, default=1.2)
    p.add_argument('--control-strength', type=float, default=0.9)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=12)
    p.add_argument('--control-image')
    p.add_argument('--prepare', action='store_true')
    a = p.parse_args()
    if a.prepare:
        print(json.dumps(prepare_controls(a.res, threads=a.threads), indent=2))
    else:
        print(json.dumps(run_experiment(a.name, a.steps, a.control_strength, a.res,
                                       a.cfg, a.seed, a.threads,
                                       control_image=a.control_image)[0], indent=2))
