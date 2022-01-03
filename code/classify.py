import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt #导入图表库
from sklearn.metrics import roc_auc_score, confusion_matrix,accuracy_score
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import  metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import permutation_test_score
import numpy.ma as ma

def svm_classify(sample_out, flag, label, sheet):
    if flag == 0:
        df = pd.read_excel('T1C_SVM_RFE.xlsx')  # T1C_SVM_RFE\T1C_LASSO2
    else:
        df = pd.read_excel('T1C_LASSO2.xlsx')
      
    # 超参数默认值
    c = 1.0
    dg = 3
    gm = 'scale'
    
    # 分类讨论
    if label == 'idh':
        if sheet == 'T1+T2+T1C':
            score = 'roc_auc'
            kn = 'rbf'
            c = 3
            gm = 0.005
        elif sheet == 'T1+T2+T1C+ADC':
            score = 'roc_auc'
            kn = 'linear'
            c= 0.03            
        elif sheet == 'T1+T2+T1C+ASL+SWI':
            score = 'roc_auc'
            kn = 'linear'
            c= 0.007
        else:
            # 'T1+T2+T1C+ADC+ASL+SWI'
            score = 'f1'
            kn = 'poly'
            c = 7
            dg = 1
            
    elif label == 'p53':
        if sheet == 'T1+T2+T1C':
            score = 'roc_auc'
            kn = 'linear'
            c = 3
        elif sheet == 'T1+T2+T1C+ADC':
            score = 'f1'
            kn = 'linear'
            c = 0.05
        elif sheet == 'T1+T2+T1C+ASL+SWI':
            score = 'f1'
            kn = 'linear'
            c = 0.1
        else:
            # 'T1+T2+T1C+ADC+ASL+SWI'
            score = 'precision'
            kn = 'rbf'
            c = 7
            gm = 0.007
    else:
        # atrx
        if sheet == 'T1+T2+T1C':
            score = 'precision'
            kn = 'linear'
            c = 0.007 
        elif sheet == 'T1+T2+T1C+ADC':
            score = 'accuracy'
            kn = 'linear'
            c = 0.07 
        elif sheet == 'T1+T2+T1C+ASL+SWI':
            score = 'accuracy'
            kn = 'linear'
            c = 0.03 
        else:
            # 'T1+T2+T1C+ADC+ASL+SWI'
            score = 'accuracy'
            kn = 'linear'
            c = 0.009 
               
    # 将dataframe的表格转成两个numpy.ndarray类型
    # label_arr、feature_arr
    label_arr = np.array(df[label].values.tolist())  # 标签数组
    df.drop([label], axis=1, inplace=True)
    feature_arr = df.values  # 特征数组
    name_list = [label]  # label+特征名列表
    for index, row in df.iteritems():  # 多个字符串转array，先转list再转array
        name_list.append(index)

    # 使用网格搜索（三折交叉检验）来调参
#     parameters = [{'kernel': ['linear'], 'C': [0.001, 0.01,0.1,1]},
#                   {'kernel': ['rbf'], 'C': [0.001,0.01, 0.1, 1], 'gamma': [0.01, 0.1, 1,10]},
#                   {'kernel': ['poly'], 'C': [0.001, 0.01, 0.1, 1], 'degree': [1,2]}]
#     if  label == 'idh':
#         # bingli idh
#         parameters = [{'kernel': ['linear'], 'C': [0.001,0.003,0.005,0.007,0.009, 0.01,0.03, 0.05, 0.07,0.09, 0.1, 0.3, 0.5,0.7,0.9, 1,3,5,7]},
#                           {'kernel': ['rbf'], 'C': [0.1,0.3,0.5,0.7,0.9, 1, 3, 5,7,9,10], 'gamma': [0.0001,0.0003, 0.0005, 0.0007,0.0009,0.001, 0.005, 0.007,0.009,0.01, 0.03,0.05,0.07,0.1, 0.3, 0.5,0.7,1]},
#                           {'kernel': ['poly'], 'C': [0.01, 0.03,0.05,0.07, 0.1, 0.3,0.5, 0.7,0.9, 1, 3, 5,7,10], 'degree': [1,2]}]
#     elif label == 'p53':
#         # bingli p53
#         parameters = [{'kernel': ['linear'], 'C': [0.005,0.007, 0.009,0.01,0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1,3, 5]},
#                           {'kernel': ['rbf'], 'C': [0.1,0.3,0.5,0.7,0.9, 1, 3, 5,7,9,10], 'gamma': [0.0005,0.0007,0.0009,0.001,0.003, 0.005, 0.007,0.01, 0.03,0.05,0.07,0.1, 0.3, 0.5,0.7,0.9,1]},
#                           {'kernel': ['poly'], 'C': [0.01,0.05,0.07, 0.09,0.1, 0.3, 0.7, 1], 'degree': [1,2]}]
#     else:   
#         # bingli atrx
#         parameters = [{'kernel': ['linear'], 'C': [0.003, 0.005,0.007, 0.009, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.7, 1]},
#                           {'kernel': ['rbf'], 'C': [0.3, 0.5,0.7,0.9, 1, 3, 5], 'gamma': [0.01,0.03,0.05,0.07,0.09,0.1, 0.3, 0.5,1]},
#                           {'kernel': ['poly'], 'C': [0.3, 0.5, 0.7, 0.9, 1, 3, 5], 'degree': [1,2]}]
    
   
    # parameters = {'kernel':['rbf'], 'C':np.arange(1,15,1), 'gamma':np.arange(0.001,0.02,0.005)}
    svr = svm.SVC(gamma=gm, C=c, kernel=kn, degree=dg)#gamma='scale', class_weight='balanced'
#     grid = GridSearchCV(estimator=svr, param_grid=parameters,cv=StratifiedKFold(5), scoring=score)#, scoring='roc_auc'，iid
#     grid.fit(feature_arr, label_arr)
    svr.fit(feature_arr, label_arr)
#     print('SVM best parameters: ', grid.best_params_)
    
    # Mean cross-validated score of the best_estimator
#     best_score = grid.best_score_
#     sample_out.append(round(best_score, 3))

#     best_estimator = grid.best_estimator_
    best_estimator = svr
    # 计算置换检验的p值
    score, permutation_scores, pvalue = permutation_test_score(
        best_estimator, feature_arr, label_arr, scoring=score, cv=StratifiedKFold(5), n_permutations=1000)
    print("Classification score %s (pvalue : %s)" % (score, pvalue))
    sample_out.append(round(pvalue, 3))


    #利用最优的SVM模型在训练集上再训练一遍(refit =True默认)
#     train_acc = grid.score(feature_arr, label_arr)
    y_pred1 = svr.predict(feature_arr)
    train_acc = accuracy_score(label_arr, y_pred1)
    print('train_acc: ', train_acc)
    # print('train_acc1: ', train_acc1)
    sample_out.append(round(train_acc, 3))
    
    # 算法评价：准确性、AUC、敏感性、特异性
    matrix = confusion_matrix(label_arr, y_pred1)
    print("confusion matrix: ", matrix)
    index_arr = matrix.ravel()
    tn = index_arr[0]
    fp = index_arr[1]
    fn = index_arr[2]
    tp = index_arr[3]
    # print(tn, fp,fn,tp)

    # AUC值
    auc = round(roc_auc_score(label_arr, y_pred1),3)
    sample_out.append(auc)
    print("AUC: ", auc)
    # 针对数据集中的所有正例(TP+FN)而言,模型正确判断出的正例(TP)占数据集中所有正例的比例
    # print("Recall: "+str(round((tp)/(tp+fn), 3)))
    # Precision：针对模型判断出的所有正例(TP+FP)而言,其中真正例(TP)占的比例
    precision = round(tp / (tp + fp + 0.01), 3)
    sample_out.append(precision)
    print("Precision: " + str(precision))
    # Sensitivity: 实际有病而且被正确诊断出来的概率
    sensitivity = round(tp / (tp + fn + 0.01), 3)
    sample_out.append(sensitivity)
    print(("Sensitivity: " + str(sensitivity)))
    #Specificity: 实际没病而且被正确诊断的概率
    specificity = round(1 - (fp / (fp + tn + 0.01)), 3)
    sample_out.append(specificity)
    print(("Specificity: " + str(specificity)))
    
    # 测试集
    # 预测新的数据的类别
    df = pd.read_excel('T1C_test.xlsx')
    df = pd.DataFrame(df, columns=name_list)
    # 将dataframe的表格转成2个numpy.ndarray类型
    # label_arr、feature_arr
    y_test = np.array(df[label].values.tolist())  # 标签数组
    df.drop([label], axis=1, inplace=True)
    X_test = df.values  # 特征数组

    y_pred = svr.predict(X_test)

    # 算法评价：准确性、AUC、敏感性、特异性
    matrix = confusion_matrix(y_test, y_pred)
    print("confusion matrix: ", matrix)
    index_arr = matrix.ravel()
    tn = index_arr[0]
    fp = index_arr[1]
    fn = index_arr[2]
    tp = index_arr[3]
    # print(tn, fp,fn,tp)

    # Accuracy:模型判断正确的数据(TP+TN)占总数据的比例
#     test_acc = grid.score(X_test,y_test)
    test_acc = accuracy_score(y_test, y_pred)
    print("test_acc:",test_acc)
    # print("test_acc1:", test_acc1)
    sample_out.append(round(test_acc, 3))

    # AUC值
    auc = round(roc_auc_score(y_test, y_pred),3)
    sample_out.append(auc)
    print("AUC: ", auc)
    # 针对数据集中的所有正例(TP+FN)而言,模型正确判断出的正例(TP)占数据集中所有正例的比例
    # print("Recall: "+str(round((tp)/(tp+fn), 3)))
    # Precision：针对模型判断出的所有正例(TP+FP)而言,其中真正例(TP)占的比例
    precision = round(tp / (tp + fp + 0.01), 3)
    sample_out.append(precision)
    print("Precision: " + str(precision))
    # Sensitivity: 实际有病而且被正确诊断出来的概率
    sensitivity = round(tp / (tp + fn + 0.01), 3)
    sample_out.append(sensitivity)
    print(("Sensitivity: " + str(sensitivity)))
    #Specificity: 实际没病而且被正确诊断的概率
    specificity = round(1 - (fp / (fp + tn + 0.01)), 3)
    sample_out.append(specificity)
    print(("Specificity: " + str(specificity)))

#     if grid.best_params_.get('kernel') == 'linear':
#         sample_out.append('linear')
#         sample_out.append(grid.best_params_.get('C'))
#         sample_out.append('')
#         sample_out.append('')
#     elif grid.best_params_.get('kernel') == 'rbf':
#         sample_out.append('rbf')
#         sample_out.append(grid.best_params_.get('C'))
#         sample_out.append(grid.best_params_.get('gamma'))
#         sample_out.append('')
#     else:
#         sample_out.append('poly')
#         sample_out.append(grid.best_params_.get('C'))
#         sample_out.append('')
#         sample_out.append(grid.best_params_.get('degree'))

    return sample_out   

def random_forest(sample_out, label):
    # 分类讨论
    if label == 'idh':
        n = 125
        dp = 11 
        split = 6
        leaf = 3
        scoring = 'f1'
                  
    elif label == 'p53':
        n = 170
        dp = 29
        split = 6
        leaf = 19
        scoring = 'precision'
    
    else:
        # atrx
        n = 145
        dp = 11 
        split = 4
        leaf = 7
        scoring = 'accuracy'
     
    
    df = pd.read_excel('T1C_SVM_RFE.xlsx')
    # 将dataframe的表格转成两个numpy.ndarray类型
    # label_arr、feature_arr
    label_arr = np.array(df[label].values.tolist())  # 标签数组
    df.drop([label], axis=1, inplace=True)
    feature_arr = df.values  # 特征数组
    name_list = [label]  # label+特征名列表
    for index, row in df.iteritems():  # 多个字符串转array，先转list再转array
        name_list.append(index)

    sco = scoring
    
    rf = RandomForestClassifier(n_estimators = n, max_depth = dp, min_samples_split = split, min_samples_leaf = leaf)
    rf.fit(feature_arr, label_arr)
    
    #     使用网格搜索（3折交叉检验）来调参(启发式调参)
#     parameter_1 = {'n_estimators':range(150,251,5)}
#     # 随机森林
#     rf_1 = RandomForestClassifier()#max_features='sqrt'
#     grid_1 = GridSearchCV(estimator=rf_1, param_grid=parameter_1, cv=StratifiedKFold(5), scoring = sco)
#     grid_1.fit(feature_arr, label_arr)
#     print('Random Forest最优参数: ', grid_1.best_params_)
#     #
#     parameter_2 = {'max_depth':range(1,11,1), 'min_samples_split':range(4,35,2)}
#     rf_2 = grid_1.best_estimator_
#     grid_2 = GridSearchCV(estimator=rf_2, param_grid=parameter_2, cv=StratifiedKFold(5), scoring = sco)
#     grid_2.fit(feature_arr, label_arr)
#     print('Random Forest best parameters: ', grid_2.best_params_)

#     parameter_3 ={'min_samples_leaf':range(3,31,2)}
#     rf_3 = grid_2.best_estimator_
#     grid_3 = GridSearchCV(estimator=rf_3, param_grid=parameter_3, cv=StratifiedKFold(5), scoring = sco)
#     grid_3.fit(feature_arr, label_arr)
#     print('Random Forest最优参数: ', grid_3.best_params_)

     
#     best_score = grid_3.best_score_
#     sample_out.append(round(best_score, 3))
  
#     best_estimator = grid_3.best_estimator_
    best_estimator = rf
    # 计算置换检验的p值

    score, permutation_scores, pvalue = permutation_test_score(
        best_estimator, feature_arr, label_arr, scoring=sco, cv=StratifiedKFold(5), n_permutations=1000)#accuracy
    print("Classification score %s (pvalue : %s)" % (score, pvalue))
    sample_out.append(round(pvalue, 3))

    y_pred1 = rf.predict(feature_arr)
    # 训练集
    train_acc = accuracy_score(label_arr, y_pred1)
    print('train_acc: ', train_acc)
    # print('train_acc1: ', train_acc1)
    sample_out.append(round(train_acc, 3))
    
    # 算法评价：准确性、AUC、敏感性、特异性
    matrix = confusion_matrix(label_arr, y_pred1)
    print("confusion matrix: ", matrix)
    index_arr = matrix.ravel()
    tn = index_arr[0]
    fp = index_arr[1]
    fn = index_arr[2]
    tp = index_arr[3]
    # print(tn, fp,fn,tp)

    # AUC值
    auc = round(roc_auc_score(label_arr, y_pred1),3)
    sample_out.append(auc)
    print("AUC: ", auc)
    # 针对数据集中的所有正例(TP+FN)而言,模型正确判断出的正例(TP)占数据集中所有正例的比例
    # print("Recall: "+str(round((tp)/(tp+fn), 3)))
    # Precision：针对模型判断出的所有正例(TP+FP)而言,其中真正例(TP)占的比例
    precision = round(tp / (tp + fp + 0.01), 3)
    sample_out.append(precision)
    print("Precision: " + str(precision))
    # Sensitivity: 实际有病而且被正确诊断出来的概率
    sensitivity = round(tp / (tp + fn + 0.01), 3)
    sample_out.append(sensitivity)
    print(("Sensitivity: " + str(sensitivity)))
    #Specificity: 实际没病而且被正确诊断的概率
    specificity = round(1 - (fp / (fp + tn + 0.01)), 3)
    sample_out.append(specificity)
    print(("Specificity: " + str(specificity)))
    
    # 测试集
    # 预测新的数据的类别
    df = pd.read_excel('T1C_test.xlsx')
    df = pd.DataFrame(df, columns=name_list)
    # 将dataframe的表格转成三个numpy.ndarray类型
    # label_arr、feature_arr、name_arr
    y_test = np.array(df[label].values.tolist())  # 标签数组
    df.drop([label], axis=1, inplace=True)
    X_test = df.values  # 特征数组

    y_pred = rf.predict(X_test)

    # 算法评价：准确性、敏感性、特异性、AUC
    matrix = confusion_matrix(y_test, y_pred)
    print("confusion matrix: ", matrix)
    index_arr = matrix.ravel()
    tn = index_arr[0]
    fp = index_arr[1]
    fn = index_arr[2]
    tp = index_arr[3]

    # Accuracy:模型判断正确的数据(TP+TN)占总数据的比例
    # test_acc = rf_best.oob_score_#带外分数
#     test_acc = grid_2.score(X_test, y_test)
    test_acc = accuracy_score(y_test, y_pred)
    print("test_acc:", round(test_acc, 3))
    # print("test_acc1:", round(test_acc1, 3))
    sample_out.append(round(test_acc, 3))

    # AUC值
    auc = roc_auc_score(y_test, y_pred)
    sample_out.append(round(auc, 3))
    print("AUC: ", round(auc, 3))
    # 针对数据集中的所有正例(TP+FN)而言,模型正确判断出的正例(TP)占数据集中所有正例的比例
    # print("Recall: "+str(round((tp)/(tp+fn), 3)))
    # Precision：针对模型判断出的所有正例(TP+FP)而言,其中真正例(TP)占的比例
    precision = round((tp) / (tp + fp + 0.01), 3)
    sample_out.append(precision)
    print("Precision: " + str(precision))
    # Sensitivity: 实际有病而且被正确诊断出来的概率
    sensitivity = round(tp / (tp + fn + 0.01), 3)
    sample_out.append(sensitivity)
    print(("Sensitivity: " + str(sensitivity)))
    # Specificity: 实际没病而且被正确诊断的概率
    specificity = round(1 - (fp / (fp + tn + 0.01)), 3)
    sample_out.append(specificity)
    print(("Specificity: " + str(specificity)))

    print("rf all parameters:", rf.get_params())
    
#     sample_out.append(grid_1.best_params_.get('n_estimators'))
#     sample_out.append(grid_2.best_params_.get('max_depth'))
#     sample_out.append(grid_2.best_params_.get('min_samples_split'))
#     sample_out.append(grid_3.best_params_.get('min_samples_leaf'))

    return sample_out

def adaboost(sample_out, feature_all, label_all, label):
    df = pd.read_excel('T1C_SVM_RFE.xlsx')
    # 将dataframe的表格转成两个numpy.ndarray类型
    # label_arr、feature_arr
    label_arr = np.array(df[label].values.tolist())  # 标签数组
    df.drop([label], axis=1, inplace=True)
    feature_arr = df.values  # 特征数组
    name_list = [label]  # label+特征名列表
    for index, row in df.iteritems():  # 多个字符串转array，先转list再转array
        name_list.append(index)

    # 使用网格搜索（3折交叉检验）来调参
    parameter = {'learning_rate': [0.05, 0.1, 0.3, 0.5, 0.7,0.8,0.85,0.9,0.95, 1.0],
                 'n_estimators': np.arange(10, 201, 5)}
    # ada = AdaBoostClassifier()#默认分类器为决策树#class_weight="balanced"
    grid = GridSearchCV(estimator=AdaBoostClassifier(),#DecisionTreeClassifier(max_depth = 1)
                        param_grid=parameter, cv=StratifiedKFold(5))#
    grid.fit(feature_arr, label_arr)
    print('AdaBoost best parameter: ', grid.best_params_)
    
#     best_estimator = grid.best_estimator_
#     # 计算置换检验的p值
#     # 标准化： StandardScaler
#     scaler = StandardScaler()
#     feature_arr1 = scaler.fit_transform(feature_all)

#     score, permutation_scores, pvalue = permutation_test_score(
#         best_estimator, feature_arr1, label_all, scoring='accuracy', cv=StratifiedKFold(5), n_permutations=1000)#accuracy
#     print("Classification score %s (pvalue : %s)" % (score, pvalue))
#     sample_out.append(round(pvalue, 4))

    # train_acc1 = grid.score(feature_arr, label_arr)
    y_pred1 = grid.predict(feature_arr)
    train_acc = accuracy_score(label_arr, y_pred1)
    # print('train_acc1: ', round(train_acc1, 3))
    print('train_acc: ', round(train_acc, 3))
    sample_out.append(round(train_acc, 3))

    # 预测新的数据的类别
    df = pd.read_excel('T1C_test.xlsx')
    df = pd.DataFrame(df, columns=name_list)
    # 将dataframe的表格转成三个numpy.ndarray类型
    # label_arr、feature_arr、name_arr
    y_test = np.array(df[label].values.tolist())  # 标签数组
    df.drop([label], axis=1, inplace=True)
    X_test = df.values  # 特征数组

    y_pred = grid.predict(X_test)

    # 算法评价：准确性、敏感性、特异性、AUC
    matrix = confusion_matrix(y_test, y_pred)
    print("consusion matrix: ", matrix)
    index_arr = matrix.ravel()
    tn = index_arr[0]
    fp = index_arr[1]
    fn = index_arr[2]
    tp = index_arr[3]

    # Accuracy:模型判断正确的数据(TP+TN)占总数据的比例
    # test_acc = grid.score(X_test, y_test)
    test_acc = accuracy_score(y_test, y_pred)
    print("test_acc:", round(test_acc, 3))
    sample_out.append(round(test_acc, 3))

    # AUC值
    auc = roc_auc_score(y_test, y_pred)
    sample_out.append(round(auc, 3))
    print("AUC: ", round(auc, 3))
    # 针对数据集中的所有正例(TP+FN)而言,模型正确判断出的正例(TP)占数据集中所有正例的比例
    # print("Recall: "+str(round((tp)/(tp+fn), 3)))
    # Precision：针对模型判断出的所有正例(TP+FP)而言,其中真正例(TP)占的比例
    precision = round((tp) / (tp + fp), 3)
    sample_out.append(precision)
    print("Precision: " + str(precision))
    # Sensitivity: 实际有病而且被正确诊断出来的概率
    sensitivity = round(tp / (tp + fn + 0.01), 3)
    sample_out.append(sensitivity)
    print(("Sensitivity: " + str(sensitivity)))
    # Specificity: 实际没病而且被正确诊断的概率
    specificity = round(1 - (fp / (fp + tn + 0.01)), 3)
    sample_out.append(specificity)
    print(("Specificity: " + str(specificity)))

    sample_out.append(grid.best_params_.get('n_estimators'))
    sample_out.append(grid.best_params_.get('learning_rate'))

    return sample_out