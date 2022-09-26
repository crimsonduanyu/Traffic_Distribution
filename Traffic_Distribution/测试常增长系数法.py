# -*- coding: utf-8 -*-            
# @Time : 2022/9/26 16:45
# @Author: 段钰
# @EMAIL： duanyu@bjtu.edu.cn
# @FileName: 测试常增长系数法.py
# @Software: PyCharm
import numpy as np
from RatioGrowthMethod import RatioGrowthMethod


if __name__ == '__main__':
    qij = [17., 7., 4., 7., 38., 6., 4., 5., 17.]
    qij = np.array(qij)
    qij.shape = (3, 3)
    Oi = [28.,
          51.,
          26.]
    Oi = np.array(Oi)
    Oi.shape = (3, 1)
    Dj = np.array([28., 50., 27.])
    T = np.sum(Oi)
    Ui = np.array([38.6, 91.9, 36.])
    Ui.shape = (3, 1)
    eg1 = RatioGrowthMethod(qij, Oi, Dj, T, Ui=Ui, X=166.5)
    eg1.method_use('constant')
    eg1.solve()
    eg1.show_answer()
