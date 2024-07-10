import numpy as np

C = np.eye(12)

msk = np.mod(np.arange(C.shape[0]) + 1, 4) == 0
msk = (msk[None, :] | msk[:, None]) * 1e4

print()