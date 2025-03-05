import os
import numpy as np

os.environ["OPENBLAS_VERBOSE"] = "2"
os.environ["OPENBLAS_CORETYPE"] = "AVX2"  # 或其他核心
print(np.dot(np.random.rand(1000, 1000), np.random.rand(1000, 1000)))
