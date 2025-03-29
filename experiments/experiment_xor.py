import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytsetlin.data.xor import get_xor
from pytsetlin import TsetlinMachine



import numpy as np


if __name__ == "__main__":

    # x, y = get_xor(n_rows = 1000, noise_fraction=0.3)
    # xt, yt = get_xor(n_rows = 200, noise_fraction=0.0)

    # xor gate
    x = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]], dtype=np.uint8)

    y = np.array([0, 1, 1, 0], dtype=np.uint32)

    tm = TsetlinMachine(n_clauses=4, seed=32)

    tm.set_train_data(x, y)
    tm.set_eval_data(x, y)

    tm.train()

    print(tm.C)
    print(tm.W)


