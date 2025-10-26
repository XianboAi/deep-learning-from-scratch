import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
y = np.sin(x)

plt.plot(x, y)
# plt.ylim(-0.1, 1.1) # 指定y轴范围 对于本图好像没什么区别
plt.show()