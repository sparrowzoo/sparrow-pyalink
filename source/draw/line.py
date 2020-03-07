#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import numpy as np
x1 = np.linspace(0, 64, 8)  


# # 从0到64，等分8分

# In[4]:


f1 = 2 * x1  # 函数定义
f2 = [8 for _ in range(8)]  # constant is 8


plt.figure()
plt.title('其数 VS 存储空间')
plt.xlabel('基数')
# 设置坐标轴刻度
x_ticks = np.arange(0, 68, 4) #4为ticks
x_label_ticks = [('{}K'.format(x)) for x in x_ticks]
plt.xticks(x_ticks, x_label_ticks)
y_ticks = np.arange(0, 68, 4)
y_label_ticks = [('{}{}'.format(y, 'K')) for y in y_ticks]
plt.yticks(y_ticks, y_label_ticks)
plt.ylabel('存储空间 KB')

plt.plot(x1, f1, label="稀疏矩阵")
plt.grid(True)
plt.plot(x1, f2, label="位图")
plt.legend(loc="upper left")
plt.axis([0, 64, 0, 64])
plt.show()

