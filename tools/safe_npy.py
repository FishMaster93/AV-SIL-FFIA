"""
安全加载含 object 数组（pickle 存储的 dict 等）的 .npy 文件。

np.load(allow_pickle=True) 会执行任意 pickle 代码；这里只允许重建 numpy 数组 / 标量 / dtype 所需的类，
其余任何 find_class 请求都会抛异常，用于加载从网上下载的第三方特征文件。
"""
import io
import pickle

import numpy as np
from numpy.lib import format as npf

_ALLOWED = {
    ('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'),
    ('numpy.core.multiarray', 'scalar'), ('numpy._core.multiarray', 'scalar'),
    ('numpy', 'ndarray'), ('numpy', 'dtype'),
    ('numpy.core.numeric', '_frombuffer'), ('numpy._core.numeric', '_frombuffer'),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'refused to load {module}.{name}')


def load(path):
    with open(path, 'rb') as f:
        version = npf.read_magic(f)
        reader = npf.read_array_header_1_0 if version == (1, 0) else npf.read_array_header_2_0
        shape, fortran, dtype = reader(f)
        if not dtype.hasobject:
            f.seek(0)
            return np.load(f, allow_pickle=False)
        arr = _RestrictedUnpickler(io.BufferedReader(f)).load()
    return arr.item() if getattr(arr, 'shape', None) == () else arr
