"""Pinned public ONNX encoders; no training data are sent to a remote service."""
from __future__ import annotations
import hashlib
import json
import time
import urllib.request
from pathlib import Path
import numpy as np
from .data import sha256, write_json

ENCODERS = {
    'minilm': dict(repo='sentence-transformers/all-MiniLM-L6-v2',
                   revision='5641a7880f40ebf4035d05e60c5f9b7a9c272c84',
                   model='onnx/model.onnx', prefix='', license='Apache-2.0'),
    'e5': dict(repo='intfloat/e5-small-v2',
               revision='d908e3456e5b47835e008ab1bd688bdd2fe5abe1',
               model='onnx/model_qint8_avx512_vnni.onnx', prefix='query: ', license='MIT'),
}
MAX_LENGTH = 128
BATCH_SIZE = 16


def fetch(url, path, expected=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.part')
    request = urllib.request.Request(url, headers={'User-Agent': 'paper-priority-reproducible-research/3'})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        if expected and sha256(partial) != expected:
            raise ValueError(f'Download hash mismatch: {url}')
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def ensure_encoder(name, assets, download=False):
    spec = ENCODERS[name]
    folder = Path(assets) / name
    manifest_path = folder / 'asset_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['spec'] != spec:
            raise ValueError('Encoder revision/spec changed; use a new asset folder.')
        if all((folder / f).is_file() and sha256(folder / f) == digest for f, digest in manifest['sha256'].items()):
            return folder, manifest
        raise ValueError(f'Encoder assets changed or incomplete: {folder}')
    if not download:
        raise FileNotFoundError(f'Run semantic.py prepare first. Missing assets: {folder}')
    base = f"https://huggingface.co/{spec['repo']}"
    pointer_url = f"{base}/raw/{spec['revision']}/{spec['model']}"
    with urllib.request.urlopen(pointer_url, timeout=60) as response:
        pointer = response.read().decode('utf-8')
    digest = next(line.split(':', 1)[1] for line in pointer.splitlines() if line.startswith('oid sha256:'))
    files = dict(model=spec['model'], tokenizer='tokenizer.json', card='README.md', config='config.json')
    saved = dict(model='model.onnx', tokenizer='tokenizer.json', card='MODEL_CARD.md', config='config.json')
    urls = {}
    for role, remote in files.items():
        url = f"{base}/resolve/{spec['revision']}/{remote}?download=true"
        urls[saved[role]] = url
        print(f'Download pinned encoder {name}: {remote}', flush=True)
        fetch(url, folder / saved[role], digest if role == 'model' else None)
    manifest = dict(spec=spec, urls=urls, sha256={f: sha256(folder / f) for f in saved.values()})
    write_json(manifest_path, manifest)
    return folder, manifest


def mean_pool(hidden, mask):
    hidden, mask = np.asarray(hidden), np.asarray(mask)
    weighted = (hidden * mask[..., None]).sum(axis=1)
    pooled = weighted / np.maximum(mask.sum(axis=1, keepdims=True), 1)
    pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
    return pooled.astype(np.float32)


def encode_titles(titles, name, assets):
    import onnxruntime as ort
    from tokenizers import Tokenizer
    folder, _ = ensure_encoder(name, assets)
    tokenizer = Tokenizer.from_file(str(folder / 'tokenizer.json'))
    tokenizer.enable_truncation(max_length=MAX_LENGTH)
    tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(folder / 'model.onnx'), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    input_names = {item.name for item in session.get_inputs()}
    result = []
    for start in range(0, len(titles), BATCH_SIZE):
        batch = tokenizer.encode_batch([ENCODERS[name]['prefix'] + t for t in titles[start:start+BATCH_SIZE]])
        values = dict(input_ids=np.asarray([t.ids for t in batch], dtype=np.int64),
                      attention_mask=np.asarray([t.attention_mask for t in batch], dtype=np.int64),
                      token_type_ids=np.asarray([t.type_ids for t in batch], dtype=np.int64))
        output = session.run(None, {k: values[k] for k in input_names})[0]
        if output.ndim != 3:
            raise ValueError('Expected unpooled token embeddings from ONNX encoder.')
        result.append(mean_pool(output, values['attention_mask']))
        if start % 512 == 0:
            print(f'Encode {name}: {min(start+BATCH_SIZE, len(titles))}/{len(titles)} titles', flush=True)
    return np.concatenate(result, axis=0)


def embedding_cache(rows, name, assets, cache):
    folder, manifest = ensure_encoder(name, assets)
    content = json.dumps([(r['id'], r['title']) for r in rows], ensure_ascii=False).encode('utf-8')
    key_spec = dict(rows_sha256=hashlib.sha256(content).hexdigest(), assets=manifest['sha256'],
                    encoder=ENCODERS[name], max_length=MAX_LENGTH, batch_size=BATCH_SIZE,
                    onnxruntime=__import__('onnxruntime').__version__,
                    tokenizers=__import__('tokenizers').__version__)
    key = hashlib.sha256(json.dumps(key_spec, sort_keys=True).encode()).hexdigest()
    path = Path(cache) / f'{name}_{key}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            values = saved['embeddings']
            if list(saved['ids']) != [r['id'] for r in rows] or values.shape != (len(rows), 384):
                raise ValueError('Embedding cache IDs/shape changed.')
    else:
        values = encode_titles([r['title'] for r in rows], name, assets)
        if values.shape != (len(rows), 384) or not np.all(np.isfinite(values)):
            raise ValueError('Invalid encoder output.')
        np.savez_compressed(path, ids=[r['id'] for r in rows], embeddings=values)
        write_json(path.with_suffix('.json'), key_spec)
    return values, dict(path=str(path), sha256=sha256(path), key=key_spec)
