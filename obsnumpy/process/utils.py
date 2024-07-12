
import numpy as np


def next_power_of_2(x):
    return int(1) if x == 0 else int(2**np.ceil(np.log2(x)))

