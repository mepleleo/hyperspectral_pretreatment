#!/usr/bin/env bin
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from scipy.signal import savgol_filter
from copy import deepcopy
import pywt

'''
参考博客
https://blog.csdn.net/Joseph__Lagrange/article/details/95302398
https://blog.csdn.net/Joseph__Lagrange/article/details/95302953
'''
class Pretreatment:

    def PlotSpectrum(self, spec, title='原始光谱', x=0, m=5):
        """
        :param spec: shape (n_samples, n_features)
        :return: plt
        """
        if isinstance(spec, pd.DataFrame):
            spec = spec.values
        spec = spec[:, :(spec.shape[1]-1)]
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        wl = np.linspace(x, x+(spec.shape[1]-1)*m,spec.shape[1])
        with plt.style.context(('ggplot')):
            fonts = 6
            plt.figure(figsize=(5.2, 3.1), dpi=200)
            plt.plot(wl, spec.T)
            plt.xlabel('Wavelength (nm)', fontsize=fonts)
            plt.ylabel('reabsorbance', fontsize=fonts)
            plt.title(title, fontsize=fonts)
        return plt
       

    def mean_centralization(self, sdata):
        """
        均值中心化
        """
        sdata = deepcopy(sdata)
        temp1 = np.mean(sdata, axis=0)
        temp2 = np.tile(temp1, sdata.shape[0]).reshape(
            (sdata.shape[0], sdata.shape[1]))
        return sdata - temp2

    def standardlize(self, sdata):
        """
        标准化
        """
        sdata = deepcopy(sdata)
        if isinstance(sdata, pd.DataFrame):
            sdata = sdata.values

        sdata = preprocessing.scale(sdata)
        return sdata

    def msc(self, sdata):

        sdata = deepcopy(sdata)
        if isinstance(sdata, pd.DataFrame):
            sdata = sdata.values

        n = sdata.shape[0]  # 样本数量
        k = np.zeros(sdata.shape[0])
        b = np.zeros(sdata.shape[0])


        M = np.array(np.mean(sdata, axis=0))

        from sklearn.linear_model import LinearRegression
        
        for i in range(n):
            y = sdata[i, :]
            y = y.reshape(-1, 1)
            M = M.reshape(-1, 1)
            model = LinearRegression()
            model.fit(M, y)
            k[i] = model.coef_
            b[i] = model.intercept_

        spec_msc = np.zeros_like(sdata)
        for i in range(n):
            bb = np.repeat(b[i], sdata.shape[1])
            kk = np.repeat(k[i], sdata.shape[1])
            temp = (sdata[i, :] - bb) / kk
            spec_msc[i, :] = temp
        return spec_msc

    # def msc(self, data_x):
    #
    #     absorbances = data_x.columns.values
    #     from sklearn.linear_model import LinearRegression
    #     ## 计算平均光谱做为标准光谱
    #     mean = np.mean(data_x,axis = 0)
    #
    #     n,p = data_x.shape
    #     msc_x = np.ones((n,p))
    #
    #     for i in range(n):
    #         y = data_x.values[i,:]
    #         lin = LinearRegression()
    #         lin.fit(mean.reshape(-1,1),y.reshape(-1,1))
    #         k = lin.coef_
    #         b = lin.intercept_
    #         msc_x[i,:] = (y - b) / k
    #
    #     msc_x = DataFrame(msc_x, columns=absorbances)
    #     return msc_x

    def D1(self, sdata):
        """
        一阶差分
        """
        sdata = deepcopy(sdata)
        if isinstance(sdata, pd.DataFrame):
            sdata = sdata.values
        temp1 = pd.DataFrame(sdata)
        temp2 = temp1.diff(axis=1)
        temp3 = temp2.values
        return np.delete(temp3, 0, axis=1)

    def D2(self, sdata):
        """
        二阶差分
        """
        sdata = deepcopy(sdata)
        if isinstance(sdata, pd.DataFrame):
            sdata = sdata.values
        temp2 = (pd.DataFrame(sdata)).diff(axis=1)
        temp3 = np.delete(temp2.values, 0, axis=1)
        temp4 = (pd.DataFrame(temp3)).diff(axis=1)
        spec_D2 = np.delete(temp4.values, 0, axis=1)
        return spec_D2

    def snv(self, sdata):
        """
        标准正态变量变换
        """
        sdata = deepcopy(sdata)
        if isinstance(sdata, pd.DataFrame):
            sdata = sdata.values
        temp1 = np.mean(sdata, axis=1)   # 求均值
        temp2 = np.tile(temp1, sdata.shape[1]).reshape((sdata.shape[0],
                                                        sdata.shape[1]), order='F')
        temp3 = np.std(sdata, axis=1)    # 标准差
        temp4 = np.tile(temp3, sdata.shape[1]).reshape((sdata.shape[0],
                                                        sdata.shape[1]), order='F')
        return (sdata - temp2) / temp4
    
    def max_min_normalization(self, data):
        """
        最大最小归一化
        """
        data = deepcopy(data)
        # min = np.min(data, axis=0)
        # max = np.max(data, axis=0)
        # res = (data - min) / (max - min)
        if isinstance(data, pd.DataFrame):
            data = data.values
        min_max_scaler = preprocessing.MinMaxScaler()
        res = min_max_scaler.fit_transform(data.T)
        return res.T

    def vector_normalization(self, data):
        """
        矢量归一化
        """
        data = deepcopy(data)
        if isinstance(data, pd.DataFrame):
            data = data.values
        x_mean = np.mean(data, axis=1)   # 求均值
        x_means = np.tile(x_mean, data.shape[1]).reshape((data.shape[0], data.shape[1]), order='F')
        x_2 = np.power(data,2)
        x_sum = np.sum(x_2,axis=1)
        x_sqrt = np.sqrt(x_sum)
        x_low = np.tile(x_sqrt, data.shape[1]).reshape((data.shape[0],data.shape[1]), order='F')
        return (data - x_means) / x_low


    def SG(self, data, w=5, p=3, d=0):
        """
        SG平滑 
        待处理
        """
        data = deepcopy(data)
        if isinstance(data, pd.DataFrame):
            data = data.values
        # data_sg = []
        # for item in data.iterrows():
        #     # print(item[0], item[1])
        #     data_sg.append(savgol_filter(item[1], x, y, mode=mode))
        # return DataFrame(data_sg, columns=absorbances)
        # savgol_filter(X, 2 * w + 1, polyorder=p, deriv=0)
        data = savgol_filter(data, w, polyorder=p, deriv=d)
        return data

    def wave(self, data_x):  # 小波变换
        data_x = deepcopy(data_x)
        if isinstance(data_x, pd.DataFrame):
            data_x = data_x.values
        def wave_(data_x):
            w = pywt.Wavelet('db8')  # 选用Daubechies8小波
            maxlev = pywt.dwt_max_level(len(data_x), w.dec_len)
            coeffs = pywt.wavedec(data_x, 'db8', level=maxlev)
            threshold = 0.04
            for i in range(1, len(coeffs)):
                coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
            datarec = pywt.waverec(coeffs, 'db8')
            return datarec

        tmp = None
        for i in range(data_x.shape[0]):
            if (i == 0):
                tmp = wave_(data_x[i])
            else:
                tmp = np.vstack((tmp, wave_(data_x[i])))
        return tmp

    def move_avg(self,data_x, n=15, mode="valid"):
        # 滑动平均滤波
        data_x = deepcopy(data_x)
        if isinstance(data_x, pd.DataFrame):
            data_x = data_x.values
        tmp = None
        for i in range(data_x.shape[0]):
            if (i == 0):
                tmp = np.convolve(data_x[i, :], np.ones((n,)) / n, mode=mode)
            else:
                tmp = np.vstack((tmp, np.convolve(data_x[i, :], np.ones((n,)) / n, mode=mode)))
        return tmp

#def msc(input_data, reference=None):
#    ''' Perform Multiplicative scatter correction'''
#
#    # mean centre correction
#    for i in range(input_data.shape[0]):
#        input_data[i,:] -= input_data[i,:].mean()
#
#    # Get the reference spectrum. If not given, estimate it from the mean
#    if reference is None:
#        # Calculate mean
#        ref = np.mean(input_data, axis=0)
#    else:
#        ref = reference
#
#    # Define a new array and populate it with the corrected data
#    data_msc = np.zeros_like(input_data)
#    for i in range(input_data.shape[0]):
#        # Run regression
#        fit = np.polyfit(ref, input_data[i,:], 1, full=True)
#        # Apply correction
#        data_msc[i,:] = (input_data[i,:] - fit[0][1]) / fit[0][0]
#
#    return (data_msc, ref)


#def snv(input_data):
#    # Define a new array and populate it with the corrected data
#    output_data = np.zeros_like(input_data)
#    for i in range(input_data.shape[0]):
#        # Apply correction
#        output_data[i, :] = (input_data[i, :] - np.mean(input_data[i, :])) / np.std(input_data[i, :])
#
#    return output_data

if __name__ == "__main__":
    # import scipy.stats
    # import scipy.io as scio
    # from pandas import DataFrame
    #

    # print(x.values[2,:])
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # fig1 = plt.figure()
    # plt.plot(x.columns,x.values[2])
    #
    data = pd.read_excel(r"苹果.xlsx")
    #
    x = data.drop(['name'], axis=1)
    p = Pretreatment()
    sg = p.SG(x, 4*5+1,2*3,2)
    # d1 = p.D1(x)
    p.PlotSpectrum(sg)
    plt.show()
    print("ok")
