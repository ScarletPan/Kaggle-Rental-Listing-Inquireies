import re
import numpy as np

with open("result.txt", "r") as f:
    raw = "".join(f.readlines())

str_res = re.findall(pattern="logloss : 0\.[0-9]+", string=raw)
res = [float(x.split(" : ")[1]) for x in str_res]
results = {i: [] for i in range(len(res) // 5)}
for i in range(len(res)):
    results[i % (len(res) // 5)].append(res[i])
results = {i: np.mean(results[i]) for i in results}
for item in sorted(results.items(), key=lambda x: x[1]):
    print(item)