import numpy as np
import scipy.stats as ss
import pylab


def set_ch():
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


set_ch()

v = 4
xi = 0.9
x_xi = ss.chi2.ppf(xi, v)
x = np.arange(0, 15, 0.1)
yd_c = ss.chi2.pdf(x, v)
xxf = np.arange(0, x_xi, 0.1)
yyf = ss.chi2.pdf(xxf, v)
pylab.plot(x, yd_c, 'b')
pylab.fill(np.append(xxf, x_xi), np.append(yyf, 0), 'g')
pylab.text(x_xi*1.01, 0.01, '{:.4f}'.format(x_xi))
pylab.text(10, 0.16,  r'$x - \chi^2(4)$')
pylab.text(1.5, 0.08, '置信水平0.9')
pylab.show()
