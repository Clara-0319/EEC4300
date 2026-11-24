"""Atomic local artifacts and immutable experiment inputs."""
import hashlib
import json
from pathlib import Path

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix(path.suffix+'.tmp')
    pending.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    pending.replace(path)

def freeze(path, value):
    normalized=json.loads(json.dumps(value,ensure_ascii=False))
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != normalized:
            raise ValueError('Frozen inputs differ; use a new output directory: '+str(path))
    else: write(path,value)
