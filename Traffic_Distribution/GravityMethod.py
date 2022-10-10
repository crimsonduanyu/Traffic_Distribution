# -*- coding: utf-8 -*-            
# @Time : 2022/10/10 12:59
# @Author: 段钰
# @EMAIL： duanyu@bjtu.edu.cn
# @FileName: GravityMethod.py
# @Software: PyCharm
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from RatioGrowthMethod import RatioGrowthMethod


class UnconstrainedGravityMethod(object):

    def __init__(self, qij, cij_now, cij_future, Ui, Vj, step2method='average'):
        self.step2method = step2method
        self.Vj = Vj
        self.Vj.shape = (1, len(Vj))
        self.Ui = Ui
        self.Ui.shape = (len(Ui), 1)
        self.qij = qij
        self.cij_now = cij_now
        self.cij_future = cij_future
        self.intercept = None
        self.coef = None
        self.SampleData = pd.DataFrame(columns=['Sample_Point', 'qij', 'Oi', 'Dj', 'Oi*Dj', 'cij', 'y', 'x1', 'x2'],
                                       dtype=float)
        Oi = np.sum(qij, axis=1)
        Dj = np.sum(qij, axis=0)
        self.Oi = Oi
        self.Dj = Dj
        for i in range(len(Oi)):
            for j in range(len(Dj)):
                sp_name = 'i=' + str(i + 1) + ', j=' + str(j + 1)
                cache = [sp_name, qij[i][j], Oi[i], Dj[j], Oi[i] * Dj[j], cij_now[i][j], np.log(qij[i][j]),
                         np.log(Oi[i] * Dj[j]), np.log(cij_now[i][j])]
                self.SampleData.loc[i * len(Dj) + j] = cache

    def show_sampledata(self):
        print(self.SampleData)
        return

    def coefficient_calibration(self):
        def extract_array_from_df(col_name):
            vector = self.SampleData[col_name]
            vector = np.array(vector)
            return vector

        def joint_x(xlist):
            """
            x list include x from different dimensions,
            eg y=a0+a1x1+a2x2+a3x3,
            then xlist should be [x1, x2, x3]
            and this function return a multi-dimensional Vector-X list
            """
            cache = xlist.pop(0)
            for i in xlist:
                cache = np.concatenate((cache, i))
            cache.shape = (len(xlist) + 1, int(cache.size / (len(xlist) + 1)))
            cache = cache.transpose()
            return cache

        y, x1, x2 = map(extract_array_from_df, ['y', 'x1', 'x2'])
        X = joint_x([x1, x2])
        linear_model = LinearRegression()
        linear_model.fit(X, y)
        self.coef = linear_model.coef_
        self.intercept = linear_model.intercept_

    def show_coefficients(self):
        print('截距项a0:', self.intercept)
        for i in range(len(self.coef)):
            print("系数项a" + str(i + 1) + ':', self.coef[i])
        return

    def update_qij(self):
        alpha = np.exp(self.intercept)
        UiVj_mat = self.Ui * self.Vj
        beta = self.coef[0]
        gamma = self.coef[1]

        # alpha = 0.183 作业使用的非最优参数
        # beta = 1.152
        # gamma = -1.536


        self.qij = alpha * np.power(UiVj_mat, beta) * np.power(self.cij_future, gamma)
        self.Oi = np.sum(self.qij, axis=1)
        self.Oi.shape = (len(self.Oi), 1)
        self.Dj = np.sum(self.qij, axis=0)
        self.Dj.shape = (1, len(self.Dj))
        return

    def solve(self, show_coef=False, show_first_qij=False):
        self.coefficient_calibration()
        if show_coef:
            self.show_coefficients()
            pass
        self.update_qij()
        if show_first_qij:
            print('*****重力模型求得的第一个OD表******')
            print(self.qij)
            print('********************************')
        step2compute = RatioGrowthMethod(self.qij, self.Oi, self.Dj, T=np.sum(self.Oi), Ui=self.Ui, Vj=self.Vj,
                                         X=np.sum(self.Oi))
        step2compute.method_use(self.step2method)
        step2compute.change_epsilon(0.03)
        step2compute.show_progress = True
        step2compute.solve()
        step2compute.show_answer()

    pass


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
    eg1 = UnconstrainedGravityMethod(qij, cij_now, cij_future, Ui, Vj)
    eg1.solve(True, True)

    pass
