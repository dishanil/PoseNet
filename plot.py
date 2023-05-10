from matplotlib import pyplot as plt
import numpy as np

flops_vo = 2000000000
flops_io_pos = 341350
flops_io_orient = 130320

flops_io = flops_io_pos + flops_io_orient

x = np.arange(1, 100)
y = np.zeros(len(x))

for i in x:
    time = (i) * 300
    y[i-1] = (flops_vo / time) + (((time - 1) * flops_io) / time)
    print(i, y[i-1])


plt.grid()
# plt.plot(x, y)
plt.scatter(x, y, color='red', marker='o')
plt.xlabel("Camera input every ith second")
plt.ylabel("FLOPS")

plt.savefig("flops.png")

plt.show()