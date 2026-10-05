
# 1.	Calcule la somme des carrés avec une boucle Python.
import numpy as np
import time

x = np.random.normal(size=1_000_000)

# 1. Boucle Python
start = time.perf_counter()
total_loop = 0
for v in x:
    total_loop += v * v
time_loop = time.perf_counter() - start

# 2. NumPy
start = time.perf_counter()
total_numpy = np.sum(x ** 2)
time_numpy = time.perf_counter() - start

print("Boucle :", time_loop)
print("NumPy  :", time_numpy)

# 3. Filtrage
start = time.perf_counter()
count_loop = sum(1 for v in x if v > 1.5)
time_loop2 = time.perf_counter() - start

start = time.perf_counter()
count_numpy = np.sum(x > 1.5)
time_numpy2 = time.perf_counter() - start

print("Boucle filtrage :", time_loop2)
print("NumPy filtrage  :", time_numpy2)

