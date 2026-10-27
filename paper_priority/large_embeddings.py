"""Pinned CPU-friendly large encoders. Earlier encoder code stays unchanged."""
import hashlib
import json
import urllib.request
from pathlib import Path
import numpy as np
from .data import sha256,write_json
from .embeddings import fetch
from .encoders_v4 import pool_hidden

LARGE_ENCODERS={
    'e5large':dict(repo='intfloat/e5-large-v2',revision='226d08282be20fa85017f8ec740d3298b706fc8b',
                   model='onnx/model_qint8_avx512_vnni.onnx',pooling='mean',prefix='query: ',dimension=1024,
                   source='https://huggingface.co/intfloat/e5-large-v2',license='MIT'),
    'bgelarge':dict(repo='Xenova/bge-large-en-v1.5',revision='dfeef6070b90658e1b391a6940efdb0925c1de6f',
                    model='onnx/model_quantized.onnx',pooling='cls',prefix='',dimension=1024,
                    source='https://huggingface.co/BAAI/bge-large-en-v1.5',license='MIT (upstream BAAI)',
                    conversion_source='https://huggingface.co/Xenova/bge-large-en-v1.5'),
}


def ensure_large(name,assets,download=False):
    spec=LARGE_ENCODERS[name]
    folder=Path(assets)/name
    path=folder/'asset_manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text(encoding='utf-8'))
        if manifest['spec']!=spec or any(not (folder/f).is_file() or sha256(folder/f)!=h for f,h in manifest['sha256'].items()):
            raise ValueError('Large encoder assets changed.')
        return folder,manifest
    if not download:
        raise FileNotFoundError(f'Run prepare for encoder {name}.')
    base=f"https://huggingface.co/{spec['repo']}"
    with urllib.request.urlopen(f"{base}/raw/{spec['revision']}/{spec['model']}",timeout=90) as response:
        pointer=response.read().decode()
    digest=next(line.split(':',1)[1] for line in pointer.splitlines() if line.startswith('oid sha256:'))
    files={'model.onnx':spec['model'],'tokenizer.json':'tokenizer.json','MODEL_CARD.md':'README.md','config.json':'config.json'}
    urls={}
    for local,remote in files.items():
        urls[local]=f"{base}/resolve/{spec['revision']}/{remote}?download=true"
        print(f'Download {name}: {remote}',flush=True)
        fetch(urls[local],folder/local,digest if local=='model.onnx' else None)
    manifest=dict(spec=spec,urls=urls,sha256={f:sha256(folder/f) for f in files})
    write_json(path,manifest)
    return folder,manifest


def large_cache(rows,name,assets,cache):
    import onnxruntime as ort
    from tokenizers import Tokenizer
    folder,manifest=ensure_large(name,assets)
    spec=LARGE_ENCODERS[name]
    key_spec=dict(rows_sha256=hashlib.sha256(json.dumps([(r['id'],r['title']) for r in rows],ensure_ascii=False).encode()).hexdigest(),
                  spec=spec,assets=manifest['sha256'],max_length=128,batch_size=8,onnxruntime=ort.__version__,
                  tokenizers=__import__('tokenizers').__version__,cpu_threads=2,cpu_arena=False)
    key=hashlib.sha256(json.dumps(key_spec,sort_keys=True).encode()).hexdigest()
    path=Path(cache)/f'{name}_{key}.npz'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with np.load(path,allow_pickle=False) as saved:
            if list(saved['ids'])!=[r['id'] for r in rows]:
                raise ValueError('Large embedding cache IDs changed.')
            values=saved['embeddings']
    else:
        tokenizer=Tokenizer.from_file(str(folder/'tokenizer.json'))
        tokenizer.enable_padding(pad_id=0,pad_token='[PAD]')
        tokenizer.enable_truncation(max_length=128)
        options=ort.SessionOptions()
        options.intra_op_num_threads=2
        options.inter_op_num_threads=1
        options.enable_cpu_mem_arena=False
        session=ort.InferenceSession(str(folder/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        input_names={item.name for item in session.get_inputs()}
        outputs=[]
        for start in range(0,len(rows),8):
            batch=tokenizer.encode_batch([spec['prefix']+r['title'] for r in rows[start:start+8]])
            feed=dict(input_ids=np.asarray([b.ids for b in batch],dtype=np.int64),
                      attention_mask=np.asarray([b.attention_mask for b in batch],dtype=np.int64),
                      token_type_ids=np.asarray([b.type_ids for b in batch],dtype=np.int64))
            hidden=session.run(None,{k:feed[k] for k in input_names})[0]
            if hidden.ndim!=3:
                raise ValueError('Expected token-level encoder outputs.')
            outputs.append(pool_hidden(hidden,feed['attention_mask'],spec['pooling']))
            if start%256==0:
                print(f'Encode {name}: {min(start+8,len(rows))}/{len(rows)}',flush=True)
        values=np.concatenate(outputs)
        del session
        np.savez_compressed(path,ids=[r['id'] for r in rows],embeddings=values)
        write_json(path.with_suffix('.json'),key_spec)
    if values.shape!=(len(rows),1024) or not np.all(np.isfinite(values)):
        raise ValueError('Invalid large encoder output.')
    return values,dict(path=str(path),sha256=sha256(path),spec=key_spec)
