import matplotlib.pyplot as plt
import numpy as np
x1 = np.linspace(0, 64, 8)  # 从0到32，等分8分
y1 = 2 * x1  # 函数定义
y2 = [8 for _ in range(8)]  # constant is 8
plt.figure()
plt.title('cardinality VS store space')
plt.xlabel('cardinality')
# 设置坐标轴刻度
x_ticks = np.arange(0, 68, 4)
x_label_ticks = [('{}K'.format(x)) for x in x_ticks]
plt.xticks(x_ticks, x_label_ticks)
y_ticks = np.arange(0, 68, 4)
y_label_ticks = [('{}{}'.format(y, 'K')) for y in y_ticks]
plt.yticks(y_ticks, y_label_ticks)
plt.ylabel('store space KB')

plt.plot(x1, y1, label="sparse")
plt.grid(True)
plt.plot(x1, y2, label="bit map")
plt.legend(loc="upper left")
plt.axis([0, 64, 0, 64])
plt.show()

print(512*1024*1024*8)
