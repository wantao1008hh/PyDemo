import matplotlib.pyplot as plt
import matplotlib.markers
import numpy as np

ypoints = np.array([1, 3, 4, 5, 8, 9, 6, 1, 3, 4, 5, 2, 4])

plt.plot(ypoints, marker='o')
plt.show()


plt.plot([1, 2, 3], marker=matplotlib.markers.CARETDOWNBASE)
plt.show()

ypoints = np.array([6, 2, 13, 10])
#fmt 参数
#fmt 参数定义了基本格式，如标记、线条样式和颜色。
#fmt = '[marker][line][color]'
#例如 o:r，o 表示实心圆标记，: 表示虚线，r 表示颜色为红色。
plt.plot(ypoints, 'o:r')
plt.show()

ypoints = np.array([6, 2, 13, 10])
plt.plot(ypoints, marker = 'o', ms = 20, mec = '#4CAF50', mfc = '#4CAF50')
plt.show()