# Local pretrained models

This directory contains the model checkpoints, tokenizer/configuration files, and optional ONNX/OpenVINO exports. Large weights use Git LFS.

```bash
git lfs install
git clone https://github.com/duanqiaoyi137/UMSM_code.git
cd UMSM_code
git lfs pull
```

For only the checkpoints used by the Python experiments:

```bash
git lfs pull --include="models/all-mpnet-base-v2/model.safetensors,models/bert-base-uncased/model.safetensors"
```

- `all-mpnet-base-v2`: Sentence-Transformers encoder used by S-BERT configurations. Upstream: https://huggingface.co/sentence-transformers/all-mpnet-base-v2 (revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`). Its original model card and training script are preserved.
- `bert-base-uncased`: BERT checkpoint used by the ALITE configuration. Upstream: https://huggingface.co/google-bert/bert-base-uncased (revision `86b5e0934494bd15c9632b12f734a8a67f723594`).

The original upstream licenses and notices apply to these third-party pretrained models; these files are not models trained by UMSM.

`MODEL_MANIFEST.json` at the repository root records sizes and SHA256 checksums. The local MPNet checkpoint was truncated by 13,298,592 bytes; the published copy was completed from a server copy and verified against the original upstream LFS SHA256. The original local file was not modified.

Duplicate ZIP archives and Hugging Face download caches are excluded. Ordinary GitHub ZIP downloads may contain LFS pointers; use `git lfs pull` to obtain the actual weight files.
