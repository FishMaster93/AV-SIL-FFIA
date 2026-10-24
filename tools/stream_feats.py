"""
流式加载 AV-CIL 发布的大型特征文件（K-S-CI 的视觉特征单个 pickle 约 100 GB，远超内存）。

- MultiPartFile：把 split 出来的 zip 分卷（.zip.aa/.ab/...）当成一个可 seek 的文件，zipfile 直接从中读取成员，
  不需要拼接、也不需要解压到磁盘。
- load_pooled：受限的纯 Python 反序列化器（白名单同 safe_npy），每个数组在 BUILD 时立刻在 token 维上取平均，
  只保留 (C,) 向量；memo 中不保留大块原始字节，因此内存只与视频数量 × 768 成正比。
"""
import io
import os
import pickle
import zipfile

import numpy as np
from numpy.lib import format as npf

_ALLOWED = {
    ('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'),
    ('numpy.core.multiarray', 'scalar'), ('numpy._core.multiarray', 'scalar'),
    ('numpy', 'ndarray'), ('numpy', 'dtype'),
}


class MultiPartFile(io.RawIOBase):
    def __init__(self, paths):
        self.paths = list(paths)
        self.sizes = [os.path.getsize(p) for p in self.paths]
        self.starts = np.cumsum([0] + self.sizes[:-1]).tolist()
        self.total = sum(self.sizes)
        self.pos = 0
        self.fhs = [open(p, 'rb') for p in self.paths]

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.total + off}[whence]
        return self.pos

    def readinto(self, b):
        n, done = len(b), 0
        while done < n and self.pos < self.total:
            i = max(k for k, s in enumerate(self.starts) if s <= self.pos)
            fh = self.fhs[i]
            fh.seek(self.pos - self.starts[i])
            chunk = fh.read(min(n - done, self.starts[i] + self.sizes[i] - self.pos))
            if not chunk:
                break
            b[done:done + len(chunk)] = chunk
            done += len(chunk)
            self.pos += len(chunk)
        return done

    def close(self):
        for fh in self.fhs:
            fh.close()
        super().close()


class _Pooled:
    """_reconstruct 的替身：BUILD 时把数值数组池化成向量；object 数组（外层 0 维容器）保留其元素列表。"""
    __slots__ = ('vec', 'items', 'shape')

    def __setstate__(self, state):
        _, shape, dtype, fortran, raw = state
        self.shape = tuple(shape)
        if isinstance(raw, list):
            self.items, self.vec = raw, None
        else:
            a = np.frombuffer(raw, dtype=dtype).reshape(shape, order='F' if fortran else 'C')
            self.vec, self.items = a.reshape(-1, a.shape[-1]).astype(np.float32).mean(0), None


class _Memo(dict):
    @staticmethod
    def _big(v):
        return isinstance(v, (bytes, bytearray)) and len(v) > (1 << 20)

    def __setitem__(self, k, v):
        # 大块原始字节及包含它的 state 元组只用一次，不进 memo
        if self._big(v) or (isinstance(v, tuple) and any(self._big(x) for x in v)):
            v = None
        super().__setitem__(k, v)


class _Unpickler(pickle._Unpickler):
    def __init__(self, f):
        super().__init__(f)
        self.memo = _Memo()

    def find_class(self, module, name):
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f'refused to load {module}.{name}')
        if name == '_reconstruct':
            return lambda *a: _Pooled()
        return super().find_class(module, name)


def load_pooled(f, progress=0):
    """f：.npy 文件对象（object 数组，元素为 {vid: ndarray}）。返回 ({vid: (C,) float32}, 原始形状示例)。"""
    version = npf.read_magic(f)
    (npf.read_array_header_1_0 if version == (1, 0) else npf.read_array_header_2_0)(f)
    up = _Unpickler(io.BufferedReader(f, 1 << 24) if isinstance(f, io.RawIOBase) else f)
    if progress:
        orig, cnt = _Pooled.__setstate__, [0]

        def counting(self, state):
            orig(self, state)
            if self.vec is not None:
                cnt[0] += 1
                if cnt[0] % progress == 0:
                    print(f'  {cnt[0]} arrays', flush=True)
        _Pooled.__setstate__ = counting
    top = up.load()
    if progress:
        _Pooled.__setstate__ = orig
    d = top.items[0] if isinstance(top, _Pooled) else top
    shape = next(iter(d.values())).shape
    return {k: v.vec for k, v in d.items()}, shape


def open_member(parts, name):
    """从（可能分卷的）zip 中按文件名打开成员，返回流。"""
    z = zipfile.ZipFile(MultiPartFile(parts))
    real = [i for i in z.infolist() if '__MACOSX' not in i.filename]
    cands = [i for i in real if os.path.basename(i.filename) == name]
    if not cands:   # 文件名不完全一致时按模态关键字匹配（同扩展名）
        ext = os.path.splitext(name)[1]
        cands = [i for i in real if i.filename.endswith(ext) and name.split('_')[0] in os.path.basename(i.filename)]
    if len(cands) != 1:
        raise FileNotFoundError(f'{name}: {[i.filename for i in z.infolist()]}')
    return z.open(cands[0])


def extract_member(parts, name, dst, chunk=1 << 24):
    """把（分卷）zip 中的一个成员流式解压到 dst（先写 .part，完成后改名）。"""
    import shutil
    src = open_member(parts, name)
    with open(dst + '.part', 'wb') as f:
        shutil.copyfileobj(src, f, chunk)
    os.replace(dst + '.part', dst)
