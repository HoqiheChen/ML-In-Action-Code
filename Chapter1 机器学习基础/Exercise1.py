##
    # Author:GeChen
    # Date:2019/4/1
    # Description:机器学习基础，python的简单使用
##
from numpy import *
# Latest:17:45
# Description:测试代码
def Exercise1():

    # Description:随机函数生成4*4数组
    rand = random.rand(4,4)

    # Description:数组转为矩阵，求逆矩阵
    randMat = mat(rand)
    print(randMat)
    randMatI = randMat.I
    print(randMatI)
    E = randMat * randMatI;
    print(E)
    Eye = eye(4)
    print(E-Eye)

# Latest:18:03
# Description:python中函数的入口
if __name__ == "__main__":
    Exercise1()