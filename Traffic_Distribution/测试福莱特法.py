# -*- coding: utf-8 -*-            
# @Time : 2022/9/26 17:37
# @Author: 段钰
# @EMAIL： duanyu@bjtu.edu.cn
# @FileName: 测试福莱特法.py
# @Software: PyCharm
import numpy as np
from RatioGrowthMethod import RatioGrowthMethod


if __name__ == "__main__":
    #   测试福莱特法
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
    Vj = np.array([39.3, 90.3, 36.9])
    Vj.shape = (1, 3)
    eg1 = RatioGrowthMethod(qij, Oi, Dj, T, Ui=Ui, Vj=Vj, X=166.5)
    eg1.method_use('fratar')
    eg1.change_epsilon(0.03)
    eg1.show_progress = False
    eg1.solve()
    eg1.show_answer()
