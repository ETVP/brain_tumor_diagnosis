import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sklearn.linear_model import LassoCV, Lasso
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold


def preprocess_t2(s, label):
    # 数据预处理：将特征归一化或标准化
    df1 = pd.read_excel(s + '_train1.xlsx')  
    label_arr1 = np.array(df1[label].values.tolist())
    df1.drop(['file name', label], axis=1, inplace=True)
    feature_arr1 = df1.values  # 特征数组
    name_list1 = [label]
    for index, row in df1.iteritems():  # 多个字符串转array，先转list再转array
        name_list1.append(index)
    name_arr1 = np.array(name_list1)  # 特征名数组

    df2 = pd.read_excel(s + '_test1.xlsx')  #
    label_arr2 = np.array(df2[label].values.tolist())
    df2.drop(['file name', label], axis=1, inplace=True)
    feature_arr2 = df2.values  # 特征数组
    name_list2 = [label]
    for index, row in df2.iteritems():  # 多个字符串转array，先转list再转array
        name_list2.append(index)
    name_arr2 = np.array(name_list2)  # 特征名数组

    # 归一化： MinMaxScaler
    # feature_train = MinMaxScaler().fit_transform(feature_arr1)
    # feature_test = MinMaxScaler().transform(feature_arr2)
    # 标准化： StandardScaler
    scaler = StandardScaler()
    feature_train = scaler.fit_transform(feature_arr1)
    feature_test = scaler.transform(feature_arr2)

    # 将归一化或标准化的特征分别写到“T1C_train.xlsx”和“T1C_test.xlsx”中
    arr_train = np.append(label_arr1.reshape(104, 1), feature_train,
                          axis=1)  # 四舍五入
    arr_test = np.append(label_arr2.reshape(27, 1), feature_test, axis=1)
    df_train = pd.DataFrame(arr_train, columns=name_arr1)
    df_test = pd.DataFrame(arr_test, columns=name_arr2)
    df_train.to_excel('T1C_train.xlsx', index=False)
    df_test.to_excel('T1C_test.xlsx', index=False)
    return


def select_U_Test(s, sample_out, label):
    # 特征选择：Mann-Whitney U Test + LASSO + SVM-RFE
    # Mann-Whitney U Test
    drop_list = []  # 由所有将被遗弃的特征组成的列表
    df = pd.read_excel("T1C_train.xlsx")
    # drop_list = ['xiaonao']  # 由所有将被遗弃的特征组成的列表
    # df = df.drop(['xiaonao'],axis=1)
    # 按照IDH基因型分为两组：野生型为0，突变型为1
    # 按照idh类型分为两组：GBM：1； LGG：0
    df_lgg = df[df[label] < 1]  # DataFrame
    df_gbm = df[df[label] > 0]
    df_lgg = df_lgg.drop([label], axis=1)
    df_gbm = df_gbm.drop([label], axis=1)
    # print(df_lgg.shape[0])
    # print(df_gbm.shape[0])

    
    # 按列遍历iteritems():
    for index, row in df_lgg.iteritems():  # index为列名，row为列值列表
        # print("index:", index)
        feature_lgg = df_lgg[index].values.tolist()
        feature_gbm = df_gbm[index].values.tolist()
        u_statistic, p_val = stats.mannwhitneyu(feature_lgg,
                                                feature_gbm,
                                                alternative='two-sided')
        if p_val >= 0.05:  # p值小于0.05，两组数据有显著差异,留下该特征
            drop_list.append(index)

    if s.count('+') == 4:
        feature = 4207 - drop_list.__len__()
        sample_out.append(4208)
    elif s.count('+') == 2:
        feature = 2533 - drop_list.__len__()
        sample_out.append(2534)
    elif s.count('+') == 3:
        feature =  3370 - drop_list.__len__()
        sample_out.append(3371)
    elif s.count('+') == 5:
        feature = 5044 - drop_list.__len__()
        sample_out.append(5045)

    sample_out.append(feature)
    print("U test extract features: ", feature) 

    df_select = pd.read_excel("T1C_train.xlsx")
    df_select = df_select.drop(drop_list, axis=1)
    df_select.to_excel('T1C_U_Test.xlsx', index=False)
    return


# 如果Sort = True，会将系数最大的X放在最前
def pretty_print_linear(coefs, names=None, sort=False):
    if names is None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst, key=lambda x: -np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name) for coef, name in lst)


def select_Lasso(sample_out, label):
    # LASSO
    df = pd.read_excel('T1C_U_Test.xlsx')
    # 将dataframe的表格转成三个numpy.ndarray类型
    # label_arr、feature_arr、name_arr
    label_arr = np.array(df[label].values.tolist())  # 标签数组
    df.drop([label], axis=1, inplace=True)
    feature_arr = df.values  # 特征数组
    name_list = []
    for index, row in df.iteritems():  # 多个字符串转array，先转list再转array
        name_list.append(index)
    name_arr = np.array(name_list)  # 特征名数组

    # 通过交叉检验来获取最优参数alpha
    lassocv = LassoCV(cv=StratifiedKFold(5))  # 5倍交叉验证
    lassocv.fit(feature_arr, label_arr)
    alpha = lassocv.alpha_
    print('Lasso: the best alpha: ', alpha)
    sample_out.append(alpha)
#     sample_out.append(round(alpha, 4))

    n = np.sum(lassocv.coef_ != 0)
    # print(lassocv.coef_)
    print('Lasso select features: ' + str(n))
    sample_out.append(n)
    # print("Lasso model: ", 'Y = ', pretty_print_linear(lassocv.coef_,
                                                    #    name_arr))

    # 系数为0的特征索引保存在index中
    index = np.argwhere(lassocv.coef_ == 0.0)
    # print(index)
    drop_list = []
    # 以index的值作为索引x，将对应的特征名name_arr[x]加到drop_list中
    for x in np.nditer(index):
        drop_list.append(name_arr[x])
    # 删除系数为0的特征，并将剩下的特征保存到excel表格“T1C_LASSO”中
    df_select = pd.read_excel('T1C_U_Test.xlsx')
    df_select = df_select.drop(drop_list, axis=1)
    df_select.to_excel('T1C_LASSO.xlsx', index=False)
    return n


def select_SVM_RFE(sample_out, feature_list, label):
    #递归特征消除
    #通过交叉验证的方式执行RFE，以此来选择最佳数量的特征
    df = pd.read_excel('T1C_LASSO.xlsx')
    # 将dataframe的表格转成三个numpy.ndarray类型
    # label_arr、feature_arr、name_arr
    label_arr = np.array(df[label].values.tolist())  # 标签数组
    df.drop([label], axis=1, inplace=True)
    feature_arr = df.values  # 特征数组
    name_list = []
    for index, row in df.iteritems():  # 多个字符串转array，先转list再转array
        name_list.append(index)
    name_arr = np.array(name_list)  # 特征名数组

    estimator = SVR(kernel='linear')
    #step:每次迭代移除特征的数目.取值可以是(0,1)的浮点数,表示百分之多少的特征;可以是整数,表示特征个数
    selector = RFECV(estimator, step=1, cv=StratifiedKFold(5)) #
    selector = selector.fit(feature_arr, label_arr)

    print('svm-rfe feature number:{}'.format(selector.n_features_))
    #筛选特征的布尔编码
    print('selected features:{}'.format(selector.support_))
    #对特征的重要性排序,最重要的序号为1
    print('feature rank:{}'.format(selector.ranking_))

    # 将特征排名不等于1的特征索引保存在index中
    index = np.argwhere(selector.ranking_ != 1)

    drop_list = []
    # 以index的值作为索引x，将对应的特征名name_arr[x]加到drop_list中
    if index.__len__() == 0:
        df_select = pd.read_excel('T1C_LASSO.xlsx')
        df_select.to_excel('T1C_SVM_RFE.xlsx', index=False)
        feature_list.append(df_select.columns.tolist())
    else:
        for x in np.nditer(index):
            drop_list.append(name_arr[x])
        # 删除系数为0的特征，并将剩下的特征保存到excel表格“T1C_LASSO”中
        df_select = pd.read_excel('T1C_LASSO.xlsx')
        df_select = df_select.drop(drop_list, axis=1)
        df_select.to_excel('T1C_SVM_RFE.xlsx', index=False)
        feature_list.append(df_select.columns.tolist())
     
    sample_out.append(selector.n_features_)

    return feature_list


def select_Lasso1(sample_out, label, al):
    # LASSO
    # df = pd.read_excel('T1C_SVM_RFE.xlsx')
    df = pd.read_excel('T1C_U_Test.xlsx')
    # 将dataframe的表格转成三个numpy.ndarray类型
    # label_arr、feature_arr、name_arr
    label_arr = np.array(df[label].values.tolist())  # 标签数组
    df.drop([label], axis=1, inplace=True)
    feature_arr = df.values  # 特征数组
    name_list = []
    for index, row in df.iteritems():  # 多个字符串转array，先转list再转array
        name_list.append(index)
    name_arr = np.array(name_list)  # 特征名数组

    lasso = Lasso(alpha=al)
    lasso.fit(feature_arr, label_arr)
    sample_out.append(round(al, 4))

    n = np.sum(lasso.coef_ != 0)
    print('Lasso select features: ' + str(n))
    sample_out.append(n)

    # 系数为0的特征索引保存在index中
    index = np.argwhere(lasso.coef_ == 0.0)
    # print(index)
    drop_list = []
    # 以index的值作为索引x，将对应的特征名name_arr[x]加到drop_list中
    if index.__len__() == 0:
        df_select = pd.read_excel('T1C_U_Test.xlsx')
        df_select.to_excel('T1C_LASSO.xlsx', index=False)
    else:
        for x in np.nditer(index):
            drop_list.append(name_arr[x])
        # 删除系数为0的特征，并将剩下的特征保存到excel表格“T1C_LASSO”中
        df_select = pd.read_excel('T1C_U_Test.xlsx')
        df_select = df_select.drop(drop_list, axis=1)
        df_select.to_excel('T1C_LASSO.xlsx', index=False)
    return n