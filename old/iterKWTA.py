# this is to show that the better formalized model performs the same as previous algorithmic approach

# the idea that we gradually decrease the threshold and use lateral negative weights to curve the encoding population
import numpy as np
import matplotlib.pyplot as plt

N_x = 100
N_y = 500
s_w = 0.1
s_w_lat = 0.1
s_x = 0.5
x = np.random.binomial(1, s_x, size=N_x)
w = np.random.binomial(1, s_w, size=(N_y, N_x))
w_lat = np.random.binomial(1, s_w_lat, size=(N_y, N_y))

y = np.zeros(N_y, dtype=int)
z = np.zeros(N_y, dtype=int)

v = w @ x
for ti in range(np.max(v), 0, -1):
  activation = v - w_lat @ y
  z = activation >= ti
  y = np.logical_or(y, z)
  print(ti, np.count_nonzero(y), np.count_nonzero(z), np.count_nonzero(activation >= 0) )




y2 = np.zeros(N_y, dtype=int)

while max(v) > 0:
  # print('max',max(v))
  y2[v == max(v)] = 1
  # v = v - w_lat @ y
  v = v - w_lat @ (v == max(v))
  v[y2 == 1] = 0
  print('Nonzero y: ', np.count_nonzero(y2))



print('Final sparcity 1: ', np.count_nonzero(y) / N_y)
print('Final sparcity 2: ', np.count_nonzero(y2) / N_y)
