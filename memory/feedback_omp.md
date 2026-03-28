---
name: KMP_DUPLICATE_LIB_OK for FAISS + PyTorch on macOS
description: Running FAISS and PyTorch together on this Mac requires KMP_DUPLICATE_LIB_OK=TRUE
type: feedback
---

When running scripts that use both FAISS and PyTorch on this machine, prefix with `KMP_DUPLICATE_LIB_OK=TRUE` or the process crashes with:

```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

**Why:** Multiple copies of the OpenMP runtime are linked into the process (one from FAISS, one from PyTorch). This is a known macOS issue.

**How to apply:** Always use `KMP_DUPLICATE_LIB_OK=TRUE python ...` for train_two_tower.py, train_reranker.py, evaluate.py, generate_offline_demo.py, and uvicorn. The test suite runs fine because it mocks app_state.load() (which would trigger FAISS import) during API tests.
