# -*- coding: utf-8 -*-            
# @Time : 2022/10/10 16:41
# @Author: 段钰
# @EMAIL： duanyu@bjtu.edu.cn
# @FileName: 作业第7题.py
# @Software: PyCharm
from GravityMethod import UnconstrainedGravityMethod
import numpy as np


if __name__ == '__main__':
    def to_npmatrix(x):
        x = np.array(x)
        size = int(np.sqrt(len(x)))
        x.shape = (size, size)
        return x


    qij = [17, 7, 4, 7, 38, 6, 4, 5, 17]
    cij_now = [7, 17, 22, 17, 15, 23, 22, 23, 7]
    cij_future = [4, 9, 11, 9, 8, 12, 11, 12, 4]
    qij, cij_now, cij_future = map(to_npmatrix, [qij, cij_now, cij_future])
    Ui = np.array([38.6, 91.9, 36.0])
    Vj = np.array([39.3, 90.3, 36.9])
    eg1 = UnconstrainedGravityMethod(qij, cij_now, cij_future, Ui, Vj, step2method='fratar')
    eg1.solve(True, True)

    pass
