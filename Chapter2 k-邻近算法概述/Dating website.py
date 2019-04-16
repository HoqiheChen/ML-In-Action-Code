from numpy import *
import operator

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

# Latest:16:50
# Description:将文本记录转换为Numpy
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

# Latest:14:46
# Description:归一化函数，负责将所有的特征放缩到取值范围相同。
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

# Latest:15:22
# Description:测试分类器的函数。
def datingClassTest(filename):
    hoRatio = 0.09
    datingDataMat,datingLabels = file2matrix(filename)
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],5)
        print("分类器的结果为：%d ，真实的结果为： %d" %(classifierResult,datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("错误率为：%f" % (errorCount/float(numTestVecs)))

# Latest:16:25
# Description:整体函数入口
def classifyPerson(filename):
    resultList = ['不喜欢','稍微喜欢','很喜欢']
    percentTats = float(input("玩游戏的百分比？（0-100）"))
    ffMiles = float(input("获得的飞行常客里程数？"))
    iceCream = float(input("每周消费的冰激凌公升数？"))
    datingDataMat,datingLabels = file2matrix(filename)
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr - minVals)/ranges,normMat,datingLabels,3)
    print("你的态度是：",resultList[classifierResult - 1])

if __name__ == '__main__':
    filename = "./sources/datingTestSet2.txt"
    classifyPerson(filename)

