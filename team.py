"""PS-09 optimised encoder for BAAI/bge-base-en-v1.5.

Contract (bench.py): setup() is called once and is NOT timed; encode(texts) -> np.ndarray [N, 768], L2-normalised,
in input order. The same encode() is called with the whole passage list (offline) and with one query (live).

Same weights, same architecture, same maths as reference.py. What changes is HOW it runs:
  O1  FP16 on Tensor Cores, fused QKV projection, PyTorch SDPA attention, no pooler
  O2  length-aware scheduler: sort by token length, token-budget batches, pad each batch to a multiple of 8
  O3  host pipeline: tokenise + build batches in a background thread, pinned memory, async copies, one final sync
  O4  optional torch.compile of the layer stack
  O5  CLS-only last layer (keys/values for all tokens, everything else for the CLS row only)
  O7  live path: one query -> fixed length bucket -> CUDA Graph replay

Every optimisation can be switched off with environment variables, so each phase can be measured on its own
(see run_ablation.py). PS09_STAGE picks a preset; individual PS09_<FLAG> variables override it.
"""
import os
import queue
import threading

import numpy as np
import torch
import torch.nn.functional as F

MODEL = os.environ.get("PS09_MODEL", "BAAI/bge-base-en-v1.5")
MAX_LEN = 512

# Stage presets for the ablation study. Stage 0 behaves like reference.py (FP32, pad to 512, batch 32).
STAGES = {
    0: dict(fp16=0, sort=0, budget=0, overlap=0, cls_only=0, compile=0, graph=0),
    1: dict(fp16=1, sort=0, budget=0, overlap=0, cls_only=0, compile=0, graph=0),   # O1
    2: dict(fp16=1, sort=1, budget=1, overlap=0, cls_only=0, compile=0, graph=0),   # + O2
    3: dict(fp16=1, sort=1, budget=1, overlap=1, cls_only=0, compile=0, graph=1),   # + O3, O7
    4: dict(fp16=1, sort=1, budget=1, overlap=1, cls_only=1, compile=0, graph=1),   # + O5
    5: dict(fp16=1, sort=1, budget=1, overlap=1, cls_only=1, compile=1, graph=1),   # + O4
}
STAGE = int(os.environ.get("PS09_STAGE", "4"))
CFG = dict(STAGES[STAGE])
for _k in list(CFG):
    if f"PS09_{_k.upper()}" in os.environ:
        CFG[_k] = int(os.environ[f"PS09_{_k.upper()}"])
TOKEN_BUDGET = int(os.environ.get("PS09_TOKEN_BUDGET", "65536"))  # max (batch x padded length) per batch
FIXED_BATCH = 32                                                    # used when budget batching is off
CHUNK = int(os.environ.get("PS09_CHUNK", "16384"))                  # texts tokenised per producer step
LIVE_BUCKETS = (32, 64, 128)                                        # CUDA Graph lengths for single queries
DEBUG = int(os.environ.get("PS09_DEBUG", "0"))                      # 1 = record max |activation| per layer

_S = {}  # module state filled by setup()


# ----------------------------------------------------------------------------------------------- model
class Encoder:
    """BERT forward written directly on the checkpoint's weights (so it is independent of the
    transformers version and lets us fuse QKV, choose the attention kernel and trim the last layer)."""

    def __init__(self, hf_model, dtype, device):
        self.dtype, self.device = dtype, device
        m = hf_model
        e = m.embeddings
        self.word = e.word_embeddings.weight.detach().to(device, dtype)
        self.pos = e.position_embeddings.weight.detach().to(device, dtype)
        self.typ0 = e.token_type_embeddings.weight[0].detach().to(device, dtype)   # all token_type_ids are 0
        self.eln = (e.LayerNorm.weight.detach().to(device, dtype), e.LayerNorm.bias.detach().to(device, dtype))
        self.eps = float(m.config.layer_norm_eps)
        self.heads = m.config.num_attention_heads
        self.layers = []
        for L in m.encoder.layer:
            a, ao = L.attention.self, L.attention.output
            g = lambda t: t.detach().to(device, dtype).contiguous()
            self.layers.append(dict(
                wqkv=g(torch.cat([a.query.weight, a.key.weight, a.value.weight], 0)),
                bqkv=g(torch.cat([a.query.bias, a.key.bias, a.value.bias], 0)),
                wo=g(ao.dense.weight), bo=g(ao.dense.bias),
                ln1=(g(ao.LayerNorm.weight), g(ao.LayerNorm.bias)),
                wi=g(L.intermediate.dense.weight), bi=g(L.intermediate.dense.bias),
                wf=g(L.output.dense.weight), bf=g(L.output.dense.bias),
                ln2=(g(L.output.LayerNorm.weight), g(L.output.LayerNorm.bias)),
            ))
        self.d = self.word.shape[1]
        self.dh = self.d // self.heads
        self.maxabs = []

    def _ln(self, x, p):
        return F.layer_norm(x, (self.d,), p[0], p[1], self.eps)  # CUDA LN accumulates in FP32

    def _layer(self, P, h, mask, cls_only):
        B, T, D = h.shape
        H, dh = self.heads, self.dh
        if not cls_only:
            qkv = F.linear(h, P["wqkv"], P["bqkv"]).view(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            res = h
        else:  # keys/values need every token; the query and everything after it only the CLS row
            kv = F.linear(h, P["wqkv"][D:], P["bqkv"][D:]).view(B, T, 2, H, dh).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
            res = h[:, :1]
            q = F.linear(res, P["wqkv"][:D], P["bqkv"][:D]).view(B, 1, H, dh).transpose(1, 2)
        ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        ctx = ctx.transpose(1, 2).reshape(B, -1, D)
        x = self._ln(F.linear(ctx, P["wo"], P["bo"]) + res, P["ln1"])
        y = F.linear(F.gelu(F.linear(x, P["wi"], P["bi"])), P["wf"], P["bf"])  # exact (erf) GELU, as BERT
        return self._ln(y + x, P["ln2"])

    def forward(self, ids, attn, use_mask=True, cls_only=False):
        """ids, attn: [B, T] int64 on device. Returns L2-normalised CLS embeddings [B, 768] in FP32."""
        T = ids.shape[1]
        h = self.word[ids] + self.pos[:T] + self.typ0
        h = self._ln(h, self.eln)
        mask = None
        if use_mask:  # additive mask, broadcast over heads and query rows
            mask = ((1 - attn[:, None, None, :].to(self.dtype)) * torch.finfo(self.dtype).min)
        n = len(self.layers)
        for i, P in enumerate(self.layers):
            h = self._layer(P, h, mask, cls_only and i == n - 1)
            if DEBUG and not (h.is_cuda and torch.cuda.is_current_stream_capturing()):
                self.maxabs.append(float(h.abs().max()))
        cls = h[:, 0].float()                    # pooling + normalisation in FP32
        return F.normalize(cls, dim=-1)


# ----------------------------------------------------------------------------------------------- setup
def setup():
    from transformers import AutoModel, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False   # keep FP32 stages bit-comparable with the reference
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    dtype = torch.float16 if (CFG["fp16"] and cuda) else torch.float32
    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModel.from_pretrained(MODEL).float().eval()
    enc = Encoder(hf, dtype, device)
    del hf
    fast = tok.backend_tokenizer          # Rust tokenizer: batch encoding releases the GIL
    fast.enable_truncation(MAX_LEN)
    fast.no_padding()
    _S.update(tok=tok, fast=fast, enc=enc, device=device, dtype=dtype, cuda=cuda,
              pad_id=tok.pad_token_id or 0, fwd=enc.forward, graphs={})
    torch.set_grad_enabled(False)

    if CFG["compile"] and cuda:
        _S["fwd"] = torch.compile(enc.forward, dynamic=True)

    # warm-up: kernel selection, cuBLAS handles, allocator pools, compile caches
    if cuda:
        with torch.inference_mode():
            for T in (8, 16, 32, 64, 128, 256, 512):
                b = max(1, min(256, TOKEN_BUDGET // T))
                ids = torch.full((b, T), 101, device=device)
                _S["fwd"](ids, torch.ones_like(ids), True, bool(CFG["cls_only"]))
        torch.cuda.synchronize()
        if CFG["graph"]:
            _capture_graphs()


def _capture_graphs():
    """O7: one CUDA Graph per live bucket. Replay launches the whole 12-layer forward in one call."""
    enc, device = _S["enc"], _S["device"]
    for T in LIVE_BUCKETS:
        ids = torch.zeros((1, T), dtype=torch.long, device=device)
        attn = torch.zeros((1, T), dtype=torch.long, device=device)
        ids[0, 0], attn[0, 0] = 101, 1
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(3):
                enc.forward(ids, attn, True, bool(CFG["cls_only"]))
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.inference_mode():
            out = enc.forward(ids, attn, True, bool(CFG["cls_only"]))
        host_ids = torch.zeros((1, T), dtype=torch.long).pin_memory()
        host_attn = torch.zeros((1, T), dtype=torch.long).pin_memory()
        _S["graphs"][T] = (g, ids, attn, out, host_ids, host_attn)
    torch.cuda.synchronize()


# ----------------------------------------------------------------------------------------------- batching
def _pad8(n):
    return (n + 7) // 8 * 8


def _plan(lengths):
    """O2: list of index arrays. Sorted by length (longest first, so any OOM shows up at once), cut so that
    batch_size x padded_length <= TOKEN_BUDGET. Without sort/budget: input order, fixed batch of 32."""
    n = len(lengths)
    order = np.argsort(-lengths, kind="stable") if CFG["sort"] else np.arange(n)
    if not CFG["budget"]:
        return [order[i:i + FIXED_BATCH] for i in range(0, n, FIXED_BATCH)]
    batches, i = [], 0
    while i < n:
        T = _pad8(int(lengths[order[i]])) if CFG["sort"] else MAX_LEN
        b = max(1, TOKEN_BUDGET // T)
        batches.append(order[i:i + b])
        i += b
    return batches


def _build(enc_list, idx, lengths):
    """Pad one batch into pinned host tensors. Stage 0/1 pad to 512 like the reference."""
    T = _pad8(int(lengths[idx].max())) if CFG["sort"] or CFG["budget"] else MAX_LEN
    ids = np.full((len(idx), T), _S["pad_id"], dtype=np.int64)
    attn = np.zeros((len(idx), T), dtype=np.int64)
    for r, j in enumerate(idx):
        e = enc_list[j]
        ids[r, :len(e)] = e
        attn[r, :len(e)] = 1
    uniform = bool(lengths[idx].min() == T)  # no padding at all -> run attention without a mask
    ids_t, attn_t = torch.from_numpy(ids), torch.from_numpy(attn)
    if _S["cuda"]:
        ids_t, attn_t = ids_t.pin_memory(), attn_t.pin_memory()
    return ids_t, attn_t, torch.from_numpy(np.ascontiguousarray(idx)), uniform


def _tokenise(texts):
    encs = _S["fast"].encode_batch(texts)
    ids = [e.ids for e in encs]
    return ids, np.fromiter((len(x) for x in ids), dtype=np.int64, count=len(ids))


def _producer(texts, q, offset_chunks):
    """O3: runs in a thread. Tokenises chunk k+1 while the GPU works on chunk k."""
    try:
        for start in offset_chunks:
            part = texts[start:start + CHUNK]
            ids, lengths = _tokenise(part)
            for idx in _plan(lengths):
                ids_t, attn_t, idx_t, uni = _build(ids, idx, lengths)
                q.put((ids_t, attn_t, idx_t + start, uni))
        q.put(None)
    except BaseException as ex:  # surface errors in the main thread
        q.put(ex)


def _run_batch(ids_t, attn_t, uniform):
    dev = _S["device"]
    ids = ids_t.to(dev, non_blocking=True)
    attn = attn_t.to(dev, non_blocking=True)
    try:
        return _S["fwd"](ids, attn, not uniform, bool(CFG["cls_only"]))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        h = ids.shape[0] // 2
        if h == 0:
            raise
        return torch.cat([_run_batch(ids_t[:h], attn_t[:h], uniform), _run_batch(ids_t[h:], attn_t[h:], uniform)])


# ----------------------------------------------------------------------------------------------- encode
def _encode_live(text):
    """O7: single query through a captured CUDA Graph."""
    ids = _S["fast"].encode(text).ids
    T = next((b for b in LIVE_BUCKETS if b >= len(ids)), None)
    if T is None:
        return None
    g, s_ids, s_attn, s_out, h_ids, h_attn = _S["graphs"][T]
    h_ids.zero_(); h_attn.zero_()
    h_ids[0, :len(ids)] = torch.tensor(ids)
    h_attn[0, :len(ids)] = 1
    s_ids.copy_(h_ids, non_blocking=True)
    s_attn.copy_(h_attn, non_blocking=True)
    g.replay()
    return s_out.cpu().numpy()


@torch.inference_mode()
def encode(texts):
    if not _S:
        setup()
    texts = list(texts)
    N = len(texts)
    if N == 0:
        return np.zeros((0, 768), dtype=np.float32)
    if N == 1 and CFG["graph"] and _S["graphs"]:
        r = _encode_live(texts[0])
        if r is not None:
            return r

    out = torch.empty((N, 768), dtype=torch.float32, device=_S["device"])
    if CFG["overlap"]:
        q = queue.Queue(maxsize=64)
        th = threading.Thread(target=_producer, args=(texts, q, range(0, N, CHUNK)), daemon=True)
        th.start()
        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            ids_t, attn_t, idx_t, uni = item
            out[idx_t.to(_S["device"], non_blocking=True)] = _run_batch(ids_t, attn_t, uni)
        th.join()
    else:
        ids, lengths = _tokenise(texts)
        for idx in _plan(lengths):
            ids_t, attn_t, idx_t, uni = _build(ids, idx, lengths)
            out[idx_t.to(_S["device"])] = _run_batch(ids_t, attn_t, uni)
    return out.cpu().numpy()  # single device -> host copy (and the only sync)
