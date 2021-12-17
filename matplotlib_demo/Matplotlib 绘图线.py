import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([6, 2, 13, 10])
#类型	简写	说明
#'solid' (默认)	'-'	实线
#'dotted'	':'	点虚线
#'dashed'	'--'	破折线
#'dashdot'	'-.'	点划线
#'None'	'' 或 ' '	不画线
plt.plot(ypoints, linestyle = 'dotted')
plt.show()