"""Download pinned upstream artifacts, verify SHA256, and convert local GGUFs.

Run once on Windows. No image generation is performed here.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import struct
import urllib.request
import zipfile

import numpy as np

from run_pipeline import RUNTIME, executable, run_logged, sha256

ARTIFACTS = {
    'blender.zip': dict(
        url='https://download.blender.org/release/Blender4.5/blender-4.5.13-windows-x64.zip',
        size=398648740,
        sha256='b5fdf800ce65fa2f209e8f68d02667e4d720fa1c42f247c72d1882ab04decba6'),
    'sdcpp.zip': dict(
        url='https://github.com/leejet/stable-diffusion.cpp/releases/download/master-1017-be48649/sd-master-be48649-bin-win-cpu-x64.zip',
        size=18080575,
        sha256='f0f76c8176a90cafc431ca7d5049cbe8c5cea61ca7cb92b78e07361d41d8fb21'),
    'sd_turbo.safetensors': dict(
        url='https://huggingface.co/stabilityai/sd-turbo/resolve/b261bac6fd2cf515557d5d0707481eafa0485ec2/sd_turbo.safetensors',
        size=5214561328,
        sha256='3f067a1b943cf162f2b8f8588f6cf5824bd5b4c7d1d88d87164b9ca123616549'),
    'control_depth_sd21.safetensors': dict(
        url='https://huggingface.co/thibaud/controlnet-sd21/resolve/ba159a8ce47185aa1821db7fe10137cf2f5e04dd/control_v11p_sd21_depth.safetensors',
        size=728494536,
        sha256='ed33ec158b6ea3bf61494ee71adc16d45886d5f6611d108345a184ebe7d08d81'),
}


def download_verified(name, spec):
    dest = RUNTIME / 'downloads' / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file():
        partial = dest.with_suffix(dest.suffix + '.part')
        start = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(spec['url'], headers={'Range': f'bytes={start}-'} if start else {})
        print(f'Downloading {name}: resume at {start} bytes', flush=True)
        with urllib.request.urlopen(request, timeout=90) as response:
            mode = 'ab' if start and response.status == 206 else 'wb'
            with partial.open(mode) as target:
                shutil.copyfileobj(response, target, length=8 * 1024 * 1024)
        if partial.stat().st_size != spec['size'] or sha256(partial) != spec['sha256']:
            raise RuntimeError(f'Incomplete or corrupted download: {name}; rerun to resume.')
        partial.replace(dest)
    if dest.stat().st_size != spec['size'] or sha256(dest) != spec['sha256']:
        raise RuntimeError(f'Checksum failed: {name}. Move the invalid file aside and rerun.')
    print(f'Verified {name}', flush=True)
    return dest


def normalize_f16(source, output):
    """Stream F32->F16 safetensors, avoiding a native converter F16 crash.

    No training or inference: only round floating-point weights to IEEE FP16.
    The original checksum-verified model is preserved.
    """
    with source.open('rb') as src:
        header_size = struct.unpack('<Q', src.read(8))[0]
        header = json.loads(src.read(header_size))
        data_start = 8 + header_size
        offset = 0
        new_header = {}
        for name, info in header.items():
            if name == '__metadata__':
                new_header[name] = info
                continue
            if info['dtype'] not in ('F32', 'F16'):
                raise ValueError(f'Unexpected source dtype: {info["dtype"]}')
            size = info['data_offsets'][1] - info['data_offsets'][0]
            if info['dtype'] == 'F32':
                size //= 2
            new_header[name] = dict(dtype='F16', shape=info['shape'],
                                    data_offsets=[offset, offset + size])
            offset += size
        encoded = json.dumps(new_header, separators=(',', ':')).encode('utf-8')
        encoded += b' ' * (-len(encoded) % 8)
        partial = output.with_suffix('.partial.safetensors')
        with partial.open('wb') as dst:
            dst.write(struct.pack('<Q', len(encoded)))
            dst.write(encoded)
            for name, info in header.items():
                if name == '__metadata__':
                    continue
                src.seek(data_start + info['data_offsets'][0])
                remaining = info['data_offsets'][1] - info['data_offsets'][0]
                while remaining:
                    block = src.read(min(4 * 1024 * 1024, remaining))
                    if not block:
                        raise ValueError('Truncated safetensors input')
                    remaining -= len(block)
                    if info['dtype'] == 'F32':
                        data = np.frombuffer(block, dtype='<f4')
                        if not np.isfinite(data).all() or (np.abs(data) > 65504).any():
                            raise ValueError(f'F16 overflow/nonfinite source: {name}')
                        block = data.astype('<f2').tobytes()
                    dst.write(block)
        if partial.stat().st_size != 8 + len(encoded) + offset:
            raise ValueError('F16 output size mismatch')
        partial.replace(output)
    print('Normalized all source weights to F16:', output.name, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    manifest_path = RUNTIME / 'resources.json'
    if args.check_only:
        record = json.loads(manifest_path.read_text(encoding='utf-8'))
        for name, info in record['models'].items():
            p = RUNTIME / 'models' / name
            assert p.stat().st_size == info['bytes'] and sha256(p) == info['sha256'], name
        print('Model checksums match the preparation manifest.')
        return
    downloaded = {name: download_verified(name, spec) for name, spec in ARTIFACTS.items()}
    # Verified archives contain all DLLs; keep each archive's directory layout intact.
    for name, folder, expected in [
        ('blender.zip', 'blender', 'blender-4.5.13-windows-x64/blender.exe'),
        ('sdcpp.zip', 'sdcpp', 'sd-cli.exe'),
    ]:
        marker = RUNTIME / folder / '.extraction-complete'
        if not marker.exists():
            with zipfile.ZipFile(downloaded[name]) as archive:
                archive.extractall(RUNTIME / folder)
            if not (RUNTIME / folder / expected).is_file():
                raise RuntimeError(f'Unexpected archive layout: {name}')
            marker.write_text(ARTIFACTS[name]['sha256'], encoding='ascii')
    cli = executable('sdcpp')
    modeldir = RUNTIME / 'models'
    modeldir.mkdir(exist_ok=True)
    previous = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    models = {}
    for source, name, kind in [
        ('sd_turbo.safetensors', 'sd_turbo-q4_0.gguf', 'q4_0'),
        ('control_depth_sd21.safetensors', 'control_depth_sd21-fp16.gguf', 'f16'),
    ]:
        output = modeldir / name
        old = previous.get('models', {}).get(name)
        recipe = 'stream-f16-then-q4_0-v1' if kind == 'q4_0' else 'f16-v1'
        if not (old and old.get('recipe') == recipe and output.is_file()
                and sha256(output) == old['sha256']):
            conversion_input = downloaded[source]
            if kind == 'q4_0':
                # Upstream Q4 conversion leaves unsupported tensors at source
                # precision. Start from F16 so conv/bias fallback is F16, not F32.
                conversion_input = RUNTIME / 'downloads' / 'sd_turbo-f16.safetensors'
                normalize_f16(downloaded[source], conversion_input)
            partial = output.with_name(output.stem + '.partial.gguf')
            run_logged([cli, '-M', 'convert', '-m', conversion_input, '--type', kind,
                        '-t', '12', '-o', partial], RUNTIME / f'{kind}-conversion.log', cli.parent)
            with partial.open('rb') as f:
                if f.read(4) != b'GGUF':
                    raise RuntimeError(f'Invalid conversion: {name}')
            partial.replace(output)
        models[name] = dict(bytes=output.stat().st_size, sha256=sha256(output),
                            source=source, conversion_type=kind, recipe=recipe)
    record = dict(prepared_at=datetime.now().astimezone().isoformat(),
                  blender_version='4.5.13', sdcpp_commit='be48649',
                  artifacts=ARTIFACTS, models=models)
    manifest_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
