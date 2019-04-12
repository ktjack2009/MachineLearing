import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def meanX(dataX):
    return np.mean(dataX, axis=0)


def pca(XMat, k):
    average = meanX(XMat)   # 每个维度的平均值
    m, n = np.shape(XMat)
    data_adjust = []
    avgs = np.tile(average, (m, 1))     # 平均值纵向复制
    data_adjust = XMat - avgs    # 向量每个维度都减去平均值
    covX = np.cov(data_adjust.T)  # 计算协方差矩阵
    featValue, featVec = np.linalg.eig(covX)  # 求解协方差矩阵的特征值和特征向量
    index = np.argsort(-featValue)  # 依照featValue进行从大到小排序
    finalData = []
    if k > n:
        print("k must lower than feature number")
        return
    else:
        selectVec = np.matrix(featVec.T[index[:k]])  # 所以这里须要进行转置
        finalData = data_adjust * selectVec.T
        reconData = (finalData * selectVec) + average
        return finalData, reconData


def loaddata(datafile):
    return np.array(pd.read_csv(datafile, sep="\t", header=-1)).astype(np.float)


def plotBestFit(data1, data2):
    dataArr1 = np.array(data1)
    dataArr2 = np.array(data2)
    m = np.shape(dataArr1)[0]
    axis_x1 = []
    axis_y1 = []
    axis_x2 = []
    axis_y2 = []
    for i in range(m):
        axis_x1.append(dataArr1[i, 0])
        axis_y1.append(dataArr1[i, 1])
        axis_x2.append(dataArr2[i, 0])
        axis_y2.append(dataArr2[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(axis_x1, axis_y1, s=50, c='red', marker='s')
    ax.scatter(axis_x2, axis_y2, s=50, c='blue')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig("outfile.png")
    plt.show()


def main():
    datafile = "/Users/dsj/workspace/MachineLearing/BaseSkill/pca_data.txt"
    # 原数据是四维
    XMat = loaddata(datafile)
    k = 2
    x, y = pca(XMat, k)
    plotBestFit(x, y)


if __name__ == "__main__":
    main()
