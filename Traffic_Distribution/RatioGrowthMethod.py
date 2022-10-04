# -*- coding: utf-8 -*-            
# @Time : 2022/9/26 13:50
# @Author: 段钰
# @EMAIL： duanyu@bjtu.edu.cn
# @FileName: RatioGrowthMethod.py
# @Software: PyCharm


import numpy as np
import warnings

class RatioGrowthMethod(object):
    def __init__(self, qij, Oi, Dj, T, Ui=None, Vj=None, X=None):
        self.iter_counter = 0
        self.final_f = False
        self.show_progress = False
        self.Dm2 = None
        self.Om2 = None
        self.F_Dj2 = None
        self.F_Oi2 = None
        self.F_Dj = None
        self.qij_m2 = None
        self.F_Oi = None
        self.epsilon = 0.03
        self.m = 0
        self.qij_m = qij
        self.Oi = Oi
        self.Dj = Dj
        self.T = T
        self.Ui = Ui
        self.Vj = Vj
        self.X = X
        self.method = ''

    def change_epsilon(self, value):
        self.epsilon = value

    def method_use(self, method=''):
        self.method = method
        return

    def get_F_Oi(self, Oi):
        Oi.shape=(np.size(Oi),1)
        self.F_Oi = self.Ui / Oi
        #self.F_Oi.shape = (len(self.F_Oi), 1)
        return self.F_Oi

    def get_F_Dj(self, Dj):
        Dj.shape = self.Vj.shape
        self.F_Dj = self.Vj / Dj
        return self.F_Dj

    def get_f(self):
        f = 0
        FO = None
        FD = None
        if self.method == '':
            raise TypeError("方法名称没有填写,请使用method_use方法")
        elif self.method == 'constant':
            try:
                FO= self.get_F_Oi(self.Oi)
            except:
                pass
            try:
                FD = self.get_F_Dj(self.Dj)
            except:
                pass
            if (FO is not None) and (FD is not None):
                warnings.warn('常增长系数法只需要Ui和Vj其一\n'
                              'Ui和Vj同时存在，将优先使用Ui进行计算，Vj将被舍弃\n'
                              '您可以选择使用其他方法', category=None, stacklevel=1, source=None)
                f = FO
            elif FO is not None:
                f = FO
            else:
                f = FD
            self.final_f = True

        elif self.method == 'average':
            FO = self.get_F_Oi(self.Oi)
            FD = self.get_F_Dj(self.Dj)
            f = (FD+FO)/2.

        elif self.method == 'detroit':
            FO = self.get_F_Oi(self.Oi)
            FD = self.get_F_Dj(self.Dj)
            f=FD*FO*self.T/self.X

        elif self.method == 'fratar':
            FO = self.get_F_Oi(self.Oi)
            FD = self.get_F_Dj(self.Dj)
            sig1 = np.sum(self.qij_m*FD, axis=1)
            sig1.shape = FO.shape
            sig2 = np.sum(self.qij_m*FO, axis=0)
            sig2.shape = FD.shape
            Li = self.Oi/sig1 #3*1
            Lj = self.Dj/sig2
            L_mat = (Li+Lj)/2
            f = FO*FD
            f = f*L_mat

        elif self.method == 'furness':
            if self.iter_counter % 2:
                f = self.get_F_Oi(self.Oi)
                #f.shape = (3,1)
                #print("counter:",self.iter_counter,"-----------",f)
            else:
                f = self.get_F_Dj(self.Dj)
                #f.shape = (1,3)
                #print("counter:", self.iter_counter, "-----------", f)

        else:
            raise TypeError("方法名称不存在，请使用method_use方法修改")
        return f

    def solve(self, max_iter=100000):
        restrained = False
        while (restrained is False) and (self.iter_counter < max_iter):
            self.iter_counter = self.iter_counter + 1
            try:
                self.F_Oi = self.get_F_Oi(self.Oi)
                self.F_Dj = self.get_F_Dj(self.Dj)
            except:
                pass
            if not self.final_f:
                f = self.get_f()
            self.qij_m2 = self.qij_m * f
            if self.show_progress:
                print("F_Oi:", self.F_Oi)
                print("F_Dj:", self.F_Dj)
                self.show_answer()
            restrained = self.check_restrained()
            self.update_params()

    def check_restrained(self):
        flag = False
        self.Om2 = np.sum(self.qij_m2, axis=1)
        self.Om2.shape = (len(self.Om2), 1)
        self.Dm2 = np.sum(self.qij_m2, axis=0)
        try:
            self.F_Oi2 = self.get_F_Oi(self.Om2)
            if np.max(self.F_Oi2) > 1 + self.epsilon or np.min(self.F_Oi2) < 1 - self.epsilon:
                return False
            flag = True
        except:
            pass
        try:
            self.F_Dj2 = self.get_F_Dj(self.Dm2)

            if np.max(self.F_Dj2) > 1 + self.epsilon or np.min(self.F_Dj2) < 1 - self.epsilon:
                return False
            flag = True
        except:
            pass
        return flag

    def update_params(self):
        self.qij_m = self.qij_m2
        self.F_Oi = self.F_Oi2
        self.F_Dj = self.F_Dj2
        self.Oi = self.Om2
        self.Dj = self.Dm2
        self.T = np.sum(self.Oi)

    def show_answer(self):
        print("Final qij=\n", self.qij_m)
        print("Final Oi=\n", self.Oi)
        print("Final Dj=\n", self.Dj)
        print("Final T=\n\t\t", self.T)
        print("Iteration Times=\n\t\t", self.iter_counter)


if __name__ == "__main__":
    #   测试佛尼斯法
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
    eg1.method_use('furness')
    eg1.change_epsilon(0.01)
    eg1.show_progress = False
    eg1.solve(100)
    eg1.show_answer()

