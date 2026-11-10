"""Export YAMNet weights for a hand-written TF.js forward pass.

- Reads the official YAMNet Keras weights (.h5, data only).
- Folds each batch-norm (center=True, scale=False, eps=1e-4) into the conv in
  front of it, so the browser only needs conv + bias + relu.
- Quantizes every kernel to int8 with one scale per output channel.
- Writes  <out>/yamnet-weights.bin  and  <out>/yamnet-weights.json  (manifest).
- Keeps a NumPy reference forward pass (features + CNN) for verification.

Usage: python3 -I export_weights.py <yamnet.h5> <out_dir>
"""
import json
import os
import sys

import h5py
import numpy as np

EPS = 1e-4
SR = 16000
WIN, HOP, NFFT = 400, 160, 512
NMEL, FMIN, FMAX = 64, 125.0, 7500.0
PATCH, PATCH_HOP = 96, 48

LAYERS = [  # (kind, stride, filters)
    ('conv', 2, 32), ('sep', 1, 64), ('sep', 2, 128), ('sep', 1, 128),
    ('sep', 2, 256), ('sep', 1, 256), ('sep', 2, 512), ('sep', 1, 512),
    ('sep', 1, 512), ('sep', 1, 512), ('sep', 1, 512), ('sep', 1, 512),
    ('sep', 2, 1024), ('sep', 1, 1024),
]


def read_h5(path):
    out = {}
    with h5py.File(path, 'r') as f:
        def grab(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = np.array(obj)
        f.visititems(grab)
    return out


def find(w, suffix):
    assert suffix in w, suffix
    return w[suffix]


def fold(w, prefix, kernel_suffix, depthwise=False):
    k = find(w, prefix + '/' + prefix + kernel_suffix)
    beta = find(w, prefix + '/bn/' + prefix + '/bn/beta:0')
    mean = find(w, prefix + '/bn/' + prefix + '/bn/moving_mean:0')
    var = find(w, prefix + '/bn/' + prefix + '/bn/moving_variance:0')
    inv = 1.0 / np.sqrt(var + EPS)
    if depthwise:           # (3,3,C,1): channel axis 2
        k = k * inv[None, None, :, None]
    else:                   # (kh,kw,in,out): channel axis 3
        k = k * inv[None, None, None, :]
    b = beta - mean * inv
    return k.astype(np.float32), b.astype(np.float32)


def build_folded(w):
    tensors = []  # list of (name, kernel, bias, kind)
    for i, (kind, stride, filters) in enumerate(LAYERS, start=1):
        name = 'layer%d' % i
        if kind == 'conv':
            k, b = fold(w, name + '/conv', '/kernel:0')
            tensors.append((name + '/conv', k, b, 'conv', stride))
        else:
            k, b = fold(w, name + '/depthwise_conv', '/depthwise_kernel:0', True)
            tensors.append((name + '/dw', k, b, 'dw', stride))
            k, b = fold(w, name + '/pointwise_conv', '/kernel:0')
            tensors.append((name + '/pw', k, b, 'conv', 1))
    k = find(w, 'logits/logits/kernel:0').astype(np.float32)
    b = find(w, 'logits/logits/bias:0').astype(np.float32)
    tensors.append(('logits', k, b, 'dense', 1))
    return tensors


def quantize(k, kind):
    """int8, one scale per output channel (axis -1 for conv/dense, axis 2 for dw)."""
    axis = 2 if kind == 'dw' else k.ndim - 1
    moved = np.moveaxis(k, axis, -1)
    amax = np.max(np.abs(moved.reshape(-1, moved.shape[-1])), axis=0)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.round(moved / scale), -127, 127).astype(np.int8)
    deq = np.moveaxis(q.astype(np.float32) * scale, -1, axis)
    return np.moveaxis(q, -1, axis), scale, deq


# ---------------- NumPy reference forward pass ----------------

def hz_to_mel(f):
    return 1127.0 * np.log(1.0 + f / 700.0)


def mel_matrix():
    nbins = NFFT // 2 + 1
    lin = np.linspace(0.0, SR / 2.0, nbins)[1:]
    spec_mel = hz_to_mel(lin)[:, None]
    edges = np.linspace(hz_to_mel(FMIN), hz_to_mel(FMAX), NMEL + 2)
    lo, ce, up = edges[:-2], edges[1:-1], edges[2:]
    lower = (spec_mel - lo) / (ce - lo)
    upper = (up - spec_mel) / (up - ce)
    m = np.maximum(0.0, np.minimum(lower, upper))
    return np.concatenate([np.zeros((1, NMEL)), m], 0).astype(np.float32)


def pad_waveform(x):
    min_n = int(0.975 * SR)
    n = max(len(x), min_n)
    hop = int(0.48 * SR)
    hops = int(np.ceil((n - min_n) / hop))
    total = min_n + hops * hop
    return np.concatenate([x, np.zeros(total - len(x), np.float32)])


def log_mel(x):
    x = pad_waveform(x.astype(np.float32))
    nframes = 1 + (len(x) - WIN) // HOP
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN)).astype(np.float32)
    frames = np.stack([x[i * HOP:i * HOP + WIN] for i in range(nframes)]) * window
    mag = np.abs(np.fft.rfft(frames, NFFT, axis=1))
    return np.log(mag @ mel_matrix() + 0.001).astype(np.float32)


def patches(spec):
    n = 1 + (spec.shape[0] - PATCH) // PATCH_HOP
    return np.stack([spec[i * PATCH_HOP:i * PATCH_HOP + PATCH] for i in range(n)])


def same_pad(size, k, s):
    out = -(-size // s)
    total = max((out - 1) * s + k - size, 0)
    return out, total // 2, total - total // 2


def conv2d(x, k, s):  # x (H,W,C), k (kh,kw,C,O)
    kh, kw, _, o = k.shape
    H, W, C = x.shape
    oh, pt, pb = same_pad(H, kh, s)
    ow, pl, pr = same_pad(W, kw, s)
    xp = np.pad(x, ((pt, pb), (pl, pr), (0, 0)))
    cols = np.empty((oh, ow, kh, kw, C), np.float32)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j, :] = xp[i:i + s * oh:s, j:j + s * ow:s, :]
    return cols.reshape(oh * ow, -1) @ k.reshape(-1, o)


def depthwise(x, k, s):  # k (3,3,C,1)
    kh, kw, C, _ = k.shape
    H, W, _ = x.shape
    oh, pt, pb = same_pad(H, kh, s)
    ow, pl, pr = same_pad(W, kw, s)
    xp = np.pad(x, ((pt, pb), (pl, pr), (0, 0)))
    out = np.zeros((oh, ow, C), np.float32)
    for i in range(kh):
        for j in range(kw):
            out += xp[i:i + s * oh:s, j:j + s * ow:s, :] * k[i, j, :, 0]
    return out


def forward(tensors, patch):
    x = patch[:, :, None]
    emb = None
    for name, k, b, kind, s in tensors:
        if kind == 'conv':
            H, W = x.shape[:2]
            oh, ow = -(-H // s), -(-W // s)
            x = np.maximum(conv2d(x, k, s) + b, 0).reshape(oh, ow, -1)
        elif kind == 'dw':
            x = np.maximum(depthwise(x, k, s) + b, 0)
        else:
            emb = x.mean(axis=(0, 1))
            x = 1.0 / (1.0 + np.exp(-(emb @ k + b)))
    return x, emb


def main():
    h5_path, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    folded = build_folded(read_h5(h5_path))

    manifest, blobs, offset = [], [], 0
    deq_tensors = []
    for name, k, b, kind, s in folded:
        q, scale, deq = quantize(k, kind)
        deq_tensors.append((name, deq, b, kind, s))
        entry = {'name': name, 'kind': kind, 'stride': s, 'shape': list(k.shape)}
        for part, arr in (('q', q.tobytes()), ('scale', scale.tobytes()), ('bias', b.tobytes())):
            # keep float32 sections 4-byte aligned
            if offset % 4:
                pad = 4 - offset % 4
                blobs.append(b'\0' * pad)
                offset += pad
            entry[part] = [offset, len(arr)]
            blobs.append(arr)
            offset += len(arr)
        manifest.append(entry)

    with open(os.path.join(out_dir, 'yamnet-weights.bin'), 'wb') as f:
        f.write(b''.join(blobs))
    with open(os.path.join(out_dir, 'yamnet-weights.json'), 'w') as f:
        json.dump({'format': 'yamnet-int8-bnfolded-v1', 'tensors': manifest}, f)
    print('wrote', offset, 'bytes,', len(manifest), 'tensors')

    if os.environ.get('DUMP_REFERENCE'):
        np.save(os.path.join(out_dir, '_folded_f32.npy'),
                np.array([(n, k, b, kd, s) for n, k, b, kd, s in folded], dtype=object),
                allow_pickle=True)
        np.save(os.path.join(out_dir, '_folded_int8deq.npy'),
                np.array(deq_tensors, dtype=object), allow_pickle=True)


if __name__ == '__main__':
    main()
