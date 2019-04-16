from numpy import *
import operator
# Latest:15:46
# Description:创建数据集和标签
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0.0,0.0],[0.0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

# Latest:15:55
# Description:KNN算法分类器，inX为输入向量，dataSet为训练样本集，labels为标签向量，k为选择最近邻居的数目
def classify0(inX,dataSet,labels,k):

    # Description:计算距离
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()

    # Description:选择最小的K个点
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1

    # Description:排序
    sortedClassount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassount[0][0]


if __name__ == '__main__':
    group,labels = createDataSet()
    res = classify0([0.8,0.4],group,labels,3)
    print(res)

