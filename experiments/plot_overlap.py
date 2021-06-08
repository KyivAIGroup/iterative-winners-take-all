import math
import matplotlib.pyplot as plt
import numpy as np

n = 50


def p_overlap(n, ay):
    return [math.comb(ay, k) * math.comb(n - ay, ay - k) / math.comb(n, ay)
            for k in range(ay + 1)]


proba = np.full((n + 1, n + 1), fill_value=np.nan)
for i, ay in enumerate(range(n + 1)):
    p = p_overlap(n, ay)
    proba[i, :ay + 1] = -np.log(p)

H_x = 10
proba[proba > H_x] = np.nan
plt.imshow(proba)
plt.xlabel('$k$')
plt.ylabel('$a_y$')
plt.title(f"$H_x={H_x}$. Entropy $h=-\log(p(k|a_y))$")
plt.colorbar()
plt.savefig('overlap.png')
plt.show()
