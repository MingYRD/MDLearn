import copy

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

def select_j(i, m):
    j = i
    while j == i:
        j = np.int(np.random.uniform(0, m))  # 从0~m中随机选取一个数，是进行整数化的
    print("random choose index for α_2:%d" % (j))
    return j  # 由于这里返回随机数，所以后面结果 可能导致不同


def clipAlpha(aj, H, L):  # 根据我们的SVM算法中的约束条件的分析，我们对获取的aj,进行了截取操作
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

class optStruct:
    def __init__(self, data_X, data_Y, C, toler):  # 输入参数分别是数据集、类别标签、常数C用于软间隔、和容错率toler
        self.X = data_X
        self.label = data_Y
        self.C = C
        self.toler = toler  # 就是软间隔中的ε，调节最大间隔大小
        self.m = data_X.shape[0]
        self.alphas = np.zeros(self.m)  # 存放每个样本点的α值
        self.b = 0  # 存放阈值
        self.eCache = np.zeros((self.m, 2))  # 用于缓存误差，每个样本点对应一个Ei值，第一列为标识符，标志是否为有效值，第二列存放有效值


def calcEk(oS, k):

    # fxk = 0.
    # for i in range(oS.m):
    #     fxk += oS.alphas[i]*oS.label[i]*oS.X[i]
    # 根据西瓜书6.24，我们可以知道预测值如何使用α值进行求解
    fxk = (np.multiply(oS.alphas.reshape(1, oS.m), oS.label.reshape(1, oS.m)))@ (oS.X @ oS.X[k, :]) + oS.b
    # np.multiply之后还是(m,1),(oS.X@oS.X[k,:])之后是(m,1),通过转置(1,m)@(m,1)-->实数后+b即可得到预测值fx
    # 获取误差值Ek
    Ek = fxk - oS.label[k]
    return Ek


# 内循环的启发式方法，获取最大差值|Ei-Ej|对应的Ej的索引J
def selectJ(i, oS, Ei):  # 注意我们要传入第一个α对应的索引i和误差值Ei,后面会用到
    maxK = -1  # 用于保存临时最大索引
    maxDeltaE = 0  # 用于保存临时最大差值--->|Ei-Ej|
    Ej = 0  # 保存我们需要的Ej误差值

    # 重点：这里我们是把SMO最后一步（根据最新阈值b，来更新Ei)提到第一步来进行了，所以这一步是非常重要的
    oS.eCache[i] = [1, Ei]

    # 开始获取各个Ek值，比较|Ei-Ej|获取Ej的所有
    # 获取所有有效的Ek值对应的索引
    validECacheList = np.where(oS.eCache[:, 0] != 0)[0]  # 根据误差缓存中第一列非0，获取对应的有效误差值
    if len(validECacheList) > 1:  # 如果有效误差缓存长度大于1(因为包括Ei),则正常进行获取j值，否则使用selectJradn方法选取一个随机J值
        for k in validECacheList:
            if k == i:  # 相同则不处理
                continue
            # 开始计算Ek值，进行对比，获取最大差值
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:  # 更新Ej及其索引位置
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej  # 返回我们找到的第二个变量α_2的位置
    else:  # 没有有效误差缓存，则随机选取一个索引，进行返回
        j = select_j(i, oS.m)
        Ej = calcEk(oS, j)
        return j, Ej


# 实现更新Ek操作,因为除了最后我们需要更新Ei之外，我们在内循环中计算α_1与α_2时还是需要用到E1与E2，
# 因为每次的E1与E2由于上一次循环中更新了α值，所以这一次也是需要更新E1与E2值，所以单独实现一个更新Ek值的方法还是有必要的
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]  # 第一列1，表示为有效标识


# 三：实现内循环函数，相比于外循环，这里包含了主要的更新操作
def innerL(i, oS):  # 由外循环提供i值（具体选取要违背kkT<这里实现>,使用交替遍历<外循环中实现>）---提供α_1的索引
    Ei = calcEk(oS, i)  # 计算E1值，主要是为了下面KKT条件需要使用到
    # print(oS.label[i] * Ei)
    # 如果下面违背了KKT条件，则正常进行α、Ek、b的更新,重点：后面单独说明下面是否满足违反KKT条件
    if (((oS.label[i] * Ei).any() < -oS.toler) and (oS.alphas[i] < oS.C)) or \
            (((oS.label[i] * Ei).any() > oS.toler) and (oS.alphas[i] > 0)):  # 注意：对于硬间隔，我们直接和1对比，对于软间隔，我们要和1 +或- ε对比
        # 开始在内循环中，选取差值最大的α_2下标索引
        j, Ej = selectJ(i, oS, Ei)
        # 因为后面要修改α_1与α_2的值，但是后面修改阈值b的时候需要用到新旧两个值，所以我们需要在更新α值之前进行保存旧值
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()

        # 分析约束条件（是对所有α都适用），一会对我们新的α_2进行截取纠正，注意：α_1是由α_2推出的，所以不需要进行验证了。
        # 如果y_1!=y_2异号时：
        if oS.label[i] != oS.label[j]:
            L = max(0, alphaJold - alphaIold)
            H = min(oS.C, oS.C + alphaJold - alphaIold)
        else:  # 如果y_1==y_2同号时
            L = max(0, alphaJold + alphaIold - oS.C)
            H = min(oS.C, alphaJold + alphaIold)
        # 上面就是将α_j调整到L,H之间
        if L == H:  # 如果L==H，之间返回0，跳出这次循环，不进行改变(单值选择，没必要）
            return 0

        # 计算η值=k_11+k_22-2k_12
        eta = oS.X[i] @ oS.X[i] + oS.X[j] @ oS.X[j] - 2.0 * oS.X[i] @ oS.X[j]  # eta性质可以知道是>=0的，所以我们只需要判断是否为0即可
        if eta <= 0:
            print("eta <= 0")
            return 0

        # 当上面所有条件都满足以后，我们开始正式修改α_2值，并更新对应的Ek值
        oS.alphas[j] += oS.label[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)

        # 查看α_2是否有足够的变化量，如果没有足够变化量，我们直接返回，不进行下面更新α_1,注意：因为α_2变化量较小，所以我们没有必要非得把值变回原来的旧值
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print("J not move enough")
            return 0

        # 开始更新α_1值,和Ek值
        oS.alphas[i] += oS.label[i] * oS.label[j] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)

        # 开始更新阈值b,正好使用到了上面更新的Ek值
        b1 = oS.b - Ei - oS.label[i] * (oS.alphas[i] - alphaIold) * oS.X[i] @ oS.X[i] - oS.label[j] * (
                oS.alphas[j] - alphaJold) * oS.X[i] @ oS.X[j]

        b2 = oS.b - Ej - oS.label[i] * (oS.alphas[i] - alphaIold) * oS.X[i] @ oS.X[j] - oS.label[j] * (
                oS.alphas[j] - alphaJold) * oS.X[j] @ oS.X[j]

        # 根据统计学习方法中阈值b在每一步中都会进行更新，
        # 1.当新值alpha_1不在界上时(0<alpha_1<C)，b_new的计算规则为：b_new=b1
        # 2.当新值alpha_2不在界上时(0 < alpha_2 < C)，b_new的计算规则为：b_new = b2
        # 3.否则当alpha_1和alpha_2都不在界上时，b_new = 1/2(b1+b2)
        if oS.alphas[i] > 0 and oS.alphas[i] < oS.C:
            oS.b = b1
        elif oS.alphas[j] > 0 and oS.alphas[j] < oS.C:
            oS.b = b2
        else:
            oS.b = 1 / 2 * (b1 + b2)

        # 注意：这里我们应该根据b_new更新一次Ei，但是我们这里没有写，因为我们将这一步提前到了最开始，即selectJ中

        # 以上全部更新完毕，开始返回标识
        return 1
    return 0  # 没有违背KKT条件


# 四：开始外循环，由于我们在内循环中实现了KKT条件的判断，所以这里我们只需要进行交替遍历即可
# 交替遍历一种方式是在所有的数据集上进行单遍扫描，另一种是在非边界上（不在边界0或C上的值）进行单遍扫描
# 交替遍历：
# 交替是通过一个外循环来选择第一个alpha值的，并且其选择过程会在两种方式之间交替：
# 一种方式是在所有数据集上进行单遍扫描，
# 另一种方式则是在非边界alpha中实现单遍扫描，所谓非边界alpha指的是那些不等于边界0或C的alpha值。
# 对整个数据集的扫描相当容易，
# 而实现非边界alpha值的扫描时，首先需要建立这些alpha值的列表，然后对这个表进行遍历。
# 同时，该步骤会跳过那些已知不变的alpha值。
def smoP(data_X, data_Y, C, toler, maxIter):
    oS = optStruct(data_X, data_Y, C, toler)
    iter = 0
    entireSet = True  # 标志是否应该遍历整个数据集
    alphaPairsChanged = 0  # 标志一次循环中α更新的次数
    # 开始进行迭代
    # 当iter >= maxIter或者((alphaPairsChanged == 0) and not entireSet)退出循环
    # 前半个判断条件很好理解，后面的判断条件中，表示上一次循环中，是在整个数据集中遍历，并且没有α值更新过，则退出
    while iter < maxIter and ((alphaPairsChanged > 0) or entireSet):
        alphaPairsChanged = 0
        if entireSet:  # entireSet是true，则在整个数据集上进行遍历
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)  # 调用内循环
            # print("full dataset, iter: %d i:%d,pairs changed:%d" % (iter, i, alphaPairsChanged))
            iter += 1  # 无论是否更新过，我们都计算迭代一次
        else:  # 遍历非边界值
            nonBounds = np.where((oS.alphas > 0) & (oS.alphas < C))[0]  # 获取非边界值中的索引
            for i in nonBounds:  # 开始遍历
                alphaPairsChanged += innerL(i, oS)
            # print("non bound, iter: %d i:%d,pairs changed:%d" % (iter, i, alphaPairsChanged))
            iter += 1  # 无论是否更新过，我们都计算迭代一次

        # 下面实现交替遍历
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:  # 如果是在非边界上，并且α更新过。则entireSet还是False,下一次还是在非边界上进行遍历。可以认为这里是倾向于非边界遍历，因为非边界遍历的样本更符合内循环中的违反KKT条件
            entireSet = True

        # print("iteration number: %d" % iter)

    return oS.b, oS.alphas


def calcWs(alphas, data_X, data_Y):
    # 根据西瓜书6.37求W
    m, n = data_X.shape
    w = np.zeros(n)
    for i in range(m):
        w += alphas[i] * data_Y[i] * data_X[i].T

    return w
def predict(x_test, w, b):
    y_test_hat = np.zeros((x_test.shape[0], 1))
    # k = self._gx(X_test)

    for i in range(x_test.shape[0]):
        k = 0.
        for j in range(x_test.shape[1]):
            k += x_test[i, j] * w[j]
        # print(k)
        k = k + b
        if k >= 0:
            y_test_hat[i, 0] = 1
        else:
            y_test_hat[i, 0] = 0
    #     y_test_hat[i, 1] = self.sigmoid(k[i])
    # y_test_hat[:, 0] = 1.0 - y_test_hat[:, 1]
    # print(y_test_hat)
    return y_test_hat
def normalize_f(data):
    return (data - data.mean())/data.std()

data_train = np.loadtxt("../other/diabetes_train.txt")
data_test = np.loadtxt("../other/diabetes_test.txt")
x_train = data_train[:, 1:data_train.shape[1]]
y_train = data_train[:, 0:1]
x_train = normalize_f(x_train)
y = copy.deepcopy(y_train)
y[y == 0] = -1

x_test = data_test[:, 1:data_train.shape[1]]
y_test = data_test[:, 0:1]
x_test = normalize_f(x_test)

C = 1
toler = 0.001
maxIter = 1000

b, alphas = smoP(x_train, y, C, toler, maxIter)
print(b)
print(alphas)
y_pred = predict(x_test, calcWs(alphas, x_train, y), b)
print(classification_report(y_test, y_pred))
