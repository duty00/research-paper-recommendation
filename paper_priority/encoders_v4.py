"""Additional pinned encoders; preserves v3 encoder implementation unchanged."""
import hashlib
import json
import urllib.request
from pathlib import Path
import numpy as np
from .data import sha256, write_json
from .embeddings import fetch, mean_pool, embedding_cache as small_cache

EXTRA_ENCODERS = {
    'e5base': dict(repo='intfloat/e5-base-v2', revision='6d89b151ea59fac713a86ac7d7e5b684c58543bf',
                   model='onnx/model_qint8_avx512_vnni.onnx', prefix='query: ', pooling='mean',
                   dimension=768, license='MIT'),
    'bge': dict(repo='BAAI/bge-base-en-v1.5', revision='a5beb1e3e68b9ab74eb54cfd186867f64f240e1a',
                model='onnx/model.onnx', prefix='', pooling='cls', dimension=768, license='MIT'),
}


def ensure_extra(name, assets, download=False):
    spec = EXTRA_ENCODERS[name]
    folder = Path(assets)/name
    path = folder/'asset_manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest['spec'] != spec or any(not (folder/f).is_file() or sha256(folder/f) != h for f,h in manifest['sha256'].items()):
            raise ValueError('Encoder assets changed.')
        return folder, manifest
    if not download:
        raise FileNotFoundError(f'Prepare missing encoder: {name}')
    base = f"https://huggingface.co/{spec['repo']}"
    with urllib.request.urlopen(f"{base}/raw/{spec['revision']}/{spec['model']}", timeout=60) as response:
        pointer = response.read().decode()
    digest = next(s.split(':',1)[1] for s in pointer.splitlines() if s.startswith('oid sha256:'))
    files = {'model.onnx':spec['model'], 'tokenizer.json':'tokenizer.json', 'MODEL_CARD.md':'README.md', 'config.json':'config.json'}
    urls = {}
    for local, remote in files.items():
        urls[local] = f"{base}/resolve/{spec['revision']}/{remote}?download=true"
        print(f'Download {name}: {remote}', flush=True)
        fetch(urls[local], folder/local, digest if local=='model.onnx' else None)
    manifest = dict(spec=spec, urls=urls, sha256={f:sha256(folder/f) for f in files})
    write_json(path, manifest)
    return folder, manifest


def pool_hidden(hidden, mask, mode):
    if mode == 'mean':
        return mean_pool(hidden, mask)
    if mode != 'cls':
        raise ValueError('Unknown pooling.')
    pooled = np.asarray(hidden[:,0], dtype=np.float32)
    return pooled/np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)


def extra_cache(rows, name, assets, cache):
    if name == 'e5':
        return small_cache(rows, name, assets, cache)
    import onnxruntime as ort
    from tokenizers import Tokenizer
    folder, manifest = ensure_extra(name, assets)
    spec = EXTRA_ENCODERS[name]
    key_spec = dict(rows=hashlib.sha256(json.dumps([(r['id'],r['title']) for r in rows],ensure_ascii=False).encode()).hexdigest(),
                    spec=spec, assets=manifest['sha256'], batch_size=16, max_length=128,
                    onnxruntime=ort.__version__, tokenizers=__import__('tokenizers').__version__)
    key = hashlib.sha256(json.dumps(key_spec,sort_keys=True).encode()).hexdigest()
    path = Path(cache)/f'{name}_{key}.npz'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with np.load(path,allow_pickle=False) as stored:
            if list(stored['ids']) != [r['id'] for r in rows]:
                raise ValueError('Cache IDs changed.')
            values = stored['embeddings']
    else:
        tokenizer = Tokenizer.from_file(str(folder/'tokenizer.json'))
        tokenizer.enable_padding(pad_id=0,pad_token='[PAD]')
        tokenizer.enable_truncation(max_length=128)
        options = ort.SessionOptions()
        options.intra_op_num_threads=2
        options.inter_op_num_threads=1
        session = ort.InferenceSession(str(folder/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        names = {item.name for item in session.get_inputs()}
        batches=[]
        for start in range(0,len(rows),16):
            batch=tokenizer.encode_batch([spec['prefix']+r['title'] for r in rows[start:start+16]])
            feed=dict(input_ids=np.asarray([b.ids for b in batch],dtype=np.int64),
                      attention_mask=np.asarray([b.attention_mask for b in batch],dtype=np.int64),
                      token_type_ids=np.asarray([b.type_ids for b in batch],dtype=np.int64))
            hidden=session.run(None,{k:feed[k] for k in names})[0]
            if hidden.ndim != 3:
                raise ValueError('Expected token embeddings.')
            batches.append(pool_hidden(hidden, feed['attention_mask'],spec['pooling']))
            if start%256==0:
                print(f'Encode {name}: {min(start+16,len(rows))}/{len(rows)}',flush=True)
        values=np.concatenate(batches)
        np.savez_compressed(path,ids=[r['id'] for r in rows],embeddings=values)
        write_json(path.with_suffix('.json'),key_spec)
    if values.shape!=(len(rows),spec['dimension']) or not np.all(np.isfinite(values)):
        raise ValueError('Invalid embedding cache.')
    return values,dict(path=str(path),sha256=sha256(path),key=key_spec)
