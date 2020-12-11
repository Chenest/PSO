# encoding:utf-8
'''
    created by Chenest on 2020/12/10
'''

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pylab as mpl
import time
mpl.rcParams['font.sans-serif'] = ['SimHei']

seed = 42
np.random.seed(seed)

class PSO:
    def __init__(self, dimension, time, size, low, up, v_low, v_high):
        # 初始化
        self.dimension = dimension  # 变量个数
        self.time = time  # 迭代的代数
        self.size = size  # 种群大小
        self.bound = []  # 变量的约束范围
        self.bound.append(low)
        self.bound.append(up)
        self.bound = np.array(self.bound)
        self.v_low = v_low
        self.v_high = v_high
        self.x = np.zeros((self.size, self.dimension))  # 所有粒子的位置
        self.v = np.zeros((self.size, self.dimension))  # 所有粒子的速度
        self.p_best = np.zeros((self.size, self.dimension))  # 每个粒子最优的位置
        self.g_best = np.zeros((1, self.dimension))[0]  # 全局最优的位置
        self.p_bestfit = -1000000 * np.ones((self.size,))
        self.g_bestfit = -1000000 * np.ones((1,))
        self.x = np.random.uniform(np.repeat(np.expand_dims(self.bound[0], 0), self.size, axis=0),
                                   np.repeat(np.expand_dims(self.bound[1], 0), self.size, axis=0),
                                   (self.size, self.dimension))
        self.p_best = self.x.copy()
        self.p_bestfit = self.fitness(self.p_best)
        self.g_bestfit = np.max(self.p_bestfit)
        self.g_best = self.x[np.argmax(self.p_bestfit)]

    def fitness(self, x):
        sum = np.zeros(self.size)
        mul = np.ones(self.size)
        for i in range(self.dimension):
            sum += np.power(x[:, i], 2) / 4000
            mul *= np.cos(x[:, i] / np.sqrt(i + 1))
        y = - sum + mul - 1
        return y

    def update(self):
        c1 = 1.4962  # 学习因子
        c2 = 1.4962
        w = 0.7298  # 自身权重因子

        self.v = w * self.v + c1 * np.random.rand(self.size, self.dimension) * (self.p_best - self.x) + c2 * np.random.rand(self.size, self.dimension) * (self.g_best - self.x)
        self.v[self.v < self.v_low] = self.v_low
        self.v[self.v > self.v_high] = self.v_high
        self.x += self.v
        self.x = np.where(self.x > self.bound[0], self.x, self.bound[0])
        self.x = np.where(self.x < self.bound[1], self.x, self.bound[1])
        fit = self.fitness(self.x)
        u = fit > self.p_bestfit
        self.p_bestfit = np.where(u, fit, self.p_bestfit)
        self.p_best = np.where(np.repeat(np.expand_dims(u, 1), self.dimension, axis=1), self.x, self.p_best)
        maxp = np.max(self.p_bestfit)
        if maxp > self.g_bestfit:
            self.g_bestfit = maxp
            self.g_best = self.p_best[np.argmax(self.p_bestfit)]

    def pso(self):
        best = []
        start = time.time()
        for gen in range(self.time):
            self.update()
            print('g_best：{}'.format(self.g_best))
            print('g_bestfit：{}'.format(self.g_bestfit))
            best.append(self.g_bestfit)
        end = time.time()
        print(end - start)
        t = [i for i in range(self.time)]
        plt.figure()
        plt.plot(t, best, color='red')
        plt.rcParams['axes.unicode_minus'] = False
        plt.margins(0)
        plt.xlabel(u"迭代次数")  # X轴标签
        plt.ylabel(u"适应度")  # Y轴标签
        plt.title(u"迭代过程")  # 标题
        plt.show()


if __name__ == '__main__':
    times = 300
    size = 500000
    dimension = 10
    v_low = -120
    v_high = 120
    low = [-600 for _ in range(dimension)]
    up = [600 for _ in range(dimension)]
    pso = PSO(dimension, times, size, low, up, v_low, v_high)
    pso.pso()
