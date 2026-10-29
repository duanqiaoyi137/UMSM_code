"""Sentence embeddings (S-BERT all-mpnet-base-v2 by default) with an on-disk cache.

Encoding follows generate_embedding.py of the original code (mean pooling over the attention
mask, L2 normalisation, str(cell) as input) but runs in batches over unique texts.
pooling="pooler" / "cls" use the pooler output / first token instead (ALITE embeds columns
with the BERT pooler output).
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

DEFAULT_MODEL = "models/all-mpnet-base-v2"
HUB_FALLBACK = {"all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2"}
POOLINGS = ("mean", "pooler", "cls")
_SEP = "\x00"


class TextEncoder:
    def __init__(self, model: str, cache_dir: str | None, device: str = "auto", batch_size: int = 64,
                 max_length: int | None = None, logger=None, pooling: str = "mean", normalize: bool = True):
        # 输入：model、cache_dir、device 等配置参数（详见签名）；输出：None；初始化实例及校验参数；继承：TextEncoder(object)。
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}")
        self.model_ref = model
        self.cache_dir = cache_dir
        self.device_pref = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.logger = logger
        self.pooling = pooling
        self.normalize = normalize
        self._tok = self._model = None
        self._memory: dict[str, dict[str, np.ndarray]] = {}
        self.dim: int | None = None
        self.n_encoded = 0
        self.encode_time_s = 0.0
        self.load_time_s = 0.0

    @property
    def model_slug(self) -> str:
        # 输入：实例状态；输出：模型缓存命名空间；继承：TextEncoder(object)。
        slug = os.path.basename(os.path.normpath(self.model_ref)).replace("/", "_")
        if self.pooling != "mean" or not self.normalize:
            slug += f"@{self.pooling}{'' if self.normalize else '-raw'}"
        # Model basenames can coincide; truncation changes vectors as well. Never read
        # the old basename-only cache, which mixed these different encoder settings.
        identity = {"version": 2, "model": self.model_ref, "max_length": self.max_length,
                    "pooling": self.pooling, "normalize": self.normalize}
        if os.path.isdir(self.model_ref):
            identity["files"] = [(f, os.stat(os.path.join(self.model_ref, f)).st_size,
                                   os.stat(os.path.join(self.model_ref, f)).st_mtime_ns)
                                  for f in sorted(os.listdir(self.model_ref))
                                  if os.path.isfile(os.path.join(self.model_ref, f))]
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        return f"{slug}-{digest}"

    def _source(self) -> str:
        # 输入：实例状态；输出：模型目录或模型标识；继承：TextEncoder(object)。
        src = self.model_ref
        if not os.path.exists(src):
            src = HUB_FALLBACK.get(os.path.basename(os.path.normpath(src)), src)
        return src

    def _load_tokenizer(self):
        # 输入：实例状态；输出：懒加载后的 tokenizer；继承：TextEncoder(object)。
        if self._tok is None:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self._source())
        return self._tok

    def special_tokens(self) -> dict[str, str]:
        # 输入：实例状态；输出：特殊词元字典；继承：TextEncoder(object)。
        """cls / sep / eos token strings of the tokenizer ('' if undefined)."""
        tok = self._load_tokenizer()
        return {k: getattr(tok, f"{k}_token", None) or "" for k in ("cls", "sep", "eos")}

    def _load(self):
        # 输入：实例状态；输出：None；懒加载模型；继承：TextEncoder(object)。
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel

        t0 = time.perf_counter()
        src = self._source()
        self.device = "cuda" if (self.device_pref == "auto" and torch.cuda.is_available()) else (
            "cpu" if self.device_pref == "auto" else self.device_pref)
        self._load_tokenizer()
        self._model = AutoModel.from_pretrained(src).to(self.device).eval()
        self.load_time_s += time.perf_counter() - t0
        if self.logger:
            self.logger.info(f"Loaded encoder {src} on {self.device} (pooling={self.pooling})")

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        # 输入：texts；输出：文本向量数组；继承：TextEncoder(object)。
        import torch
        import torch.nn.functional as F

        self._load()
        kw = {"max_length": self.max_length} if self.max_length else {}
        enc = self._tok(texts, padding=True, truncation=True, return_tensors="pt", **kw).to(self.device)
        with torch.no_grad():
            res = self._model(**enc)
        out = res[0]
        if self.pooling == "pooler":
            emb = res.pooler_output
        elif self.pooling == "cls":
            emb = out[:, 0]
        else:
            mask = enc["attention_mask"].unsqueeze(-1).expand(out.size()).float()
            emb = torch.sum(out * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy().astype(np.float32)

    # ---- cache -------------------------------------------------------------------------------
    def _cache_file(self, namespace: str) -> str | None:
        # 输入：namespace；输出：向量缓存文件路径；继承：TextEncoder(object)。
        if not self.cache_dir:
            return None
        safe = hashlib.sha1(namespace.encode("utf-8")).hexdigest()[:16]
        return os.path.join(self.cache_dir, "embeddings", self.model_slug, f"{safe}.npz")

    def _store(self, namespace: str) -> dict[str, np.ndarray]:
        # 输入：namespace；输出：指定命名空间的向量缓存字典；继承：TextEncoder(object)。
        if namespace not in self._memory:
            store: dict[str, np.ndarray] = {}
            path = self._cache_file(namespace)
            if path and os.path.exists(path):
                data = np.load(path, allow_pickle=False)
                vectors = data["vectors"]
                if len(vectors):
                    texts = data["texts"].tobytes().decode("utf-8").split(_SEP)
                    store = dict(zip(texts, vectors))
                    self.dim = vectors.shape[1]
            self._memory[namespace] = store
        return self._memory[namespace]

    def _save(self, namespace: str) -> None:
        # 输入：namespace；输出：None；保存向量缓存；继承：TextEncoder(object)。
        path = self._cache_file(namespace)
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        store = self._memory[namespace]
        texts = np.frombuffer(_SEP.join(store.keys()).encode("utf-8"), dtype=np.uint8)
        vectors = np.stack(list(store.values())) if store else np.zeros((0, self.dim or 0), np.float32)
        tmp = path + ".tmp.npz"
        np.savez(tmp, texts=texts, vectors=vectors, namespace=np.frombuffer(namespace.encode("utf-8"), np.uint8))
        os.replace(tmp, path)

    def encode(self, texts: list[str], namespace: str = "default") -> np.ndarray:
        # 输入：texts、namespace；输出：与输入顺序一致的文本向量数组；继承：TextEncoder(object)。
        """Embeddings for `texts` (n, d); unseen unique texts are encoded and cached."""
        store = self._store(namespace)
        # batches of similar length keep padding (and CPU time) small
        todo = sorted({t for t in texts if t not in store}, key=lambda t: (len(t), t))
        if todo:
            if self.logger:
                self.logger.info(f"Encoding {len(todo)} new texts [{namespace}]")
            self._load()
            t0 = time.perf_counter()
            n_batches = (len(todo) + self.batch_size - 1) // self.batch_size
            for b, i in enumerate(range(0, len(todo), self.batch_size), 1):
                batch = todo[i:i + self.batch_size]
                for t, v in zip(batch, self._encode_batch(batch)):
                    store[t] = v
                if self.logger and (b % 10 == 0 or b == n_batches) and n_batches > 10:
                    self.logger.info(f"  encoded batch {b}/{n_batches} ({time.perf_counter() - t0:.1f}s)")
            self.encode_time_s += time.perf_counter() - t0
            self.n_encoded += len(todo)
            self.dim = next(iter(store.values())).shape[0]
            self._save(namespace)
        if not texts:
            return np.zeros((0, self.dim or 0), np.float32)
        return np.stack([store[t] for t in texts])

    def encode_map(self, texts, namespace: str = "default") -> dict[str, np.ndarray]:
        # 输入：texts、namespace；输出：文本到向量的字典；继承：TextEncoder(object)。
        """{text: embedding} for the unique texts, encoding all unseen ones in one pass."""
        unique = list(dict.fromkeys(texts))
        return dict(zip(unique, self.encode(unique, namespace))) if unique else {}


def get_encoder(ctx, model: str = DEFAULT_MODEL, device: str = "auto", batch_size: int = 64,
                max_length: int | None = None, pooling: str = "mean", normalize: bool = True) -> TextEncoder:
    # 输入：ctx、model、device、batch_size、max_length、pooling、normalize；输出：复用的 TextEncoder；继承：无（模块函数）。
    local = ctx.resolve(model)
    path = local if os.path.exists(local) else model
    key = ("encoder", path, device, max_length, pooling, normalize)
    cache_dir = ctx.cache_dir if ctx.embedding_cache else None
    return ctx.shared(key, lambda: TextEncoder(path, cache_dir, device, batch_size, max_length, ctx.logger,
                                               pooling, normalize))


def encoder_stats(ctx) -> dict:
    # 输入：ctx；输出：本轮编码耗时与文本数量统计；继承：无（模块函数）。
    encs = [v for v in ctx.resources.values() if isinstance(v, TextEncoder)]
    return {"embedding_cache": ctx.embedding_cache,
            "n_texts_encoded": sum(e.n_encoded for e in encs),
            "encode_time_s": sum(e.encode_time_s for e in encs),
            "model_load_time_s": sum(e.load_time_s for e in encs)}
