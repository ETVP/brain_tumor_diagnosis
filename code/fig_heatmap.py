import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

def dfToList(path, str):
    data1 = pd.DataFrame(pd.read_excel(path, sheet_name = str))
    list2 = []

    list1 = data1.values.tolist()[2][0]
    list1 = list1.lstrip('[')
    list1 = list1.rstrip(']')
    list1 = list1.split(',')
    list1.remove(list1[0])
    for i in range(len(list1)):
        list2.append(eval(list1[i]))

    return list2

if __name__ == "__main__":
    # 找到八个分类最优特征集的并集list_all
    path1 = "C:/Users/89493/Desktop/final_outcom/jiyin_features.xlsx"
    # list1 = dfToList(path1, '1p19q')
    # # print("list1", len(list1))
    # list2 = dfToList(path1,'idh')
    # list3 = dfToList(path1,'tert')
    # list4 = dfToList(path1,'mgmt')
    # list5 = dfToList(path1,'grade')

    path2 = "C:/Users/89493/Desktop/final_outcom/bingli_features.xlsx"
    # list6 = dfToList(path2,'p53')
    # list7 = dfToList(path2,'idh')
    # list8 = dfToList(path2,'atrx')


    # print("list_len", len(list1), len(list2),len(list3),len(list4),len(list5), len(list6), len(list7), len(list8))
    # lists = list1 + list2 + list3 + list4 + list5 + list6 + list7 + list8
    # print("lists_len", len(lists), lists)

    # list_row = list(set(lists))
    # print("------------------------------")

    # print("set len", len(list_row), list_row)

    # # 找出被剔除掉的元素
    # for element in list_row:
    #     lists.remove(element)
    # print("------------------------------")

    # print(len(lists),lists)


    # # # 定义colum
    # list_colum = ["grade+", "grade-", "1p19q+", "1p19q-", "idh+", "idh-", "tert+", "tert-", "mgmt+", "mgmt-", "idh+", "idh-", "p53+", "p53-", "atrx+", "atrx-"]
    # # list_colum = ["Grade+", "Grade-", "1p19q+", "1p19q-", "IDH+", "IDH-", "TERT+", "TERT-", "MGMT+", "MGMT-", "IDH+", "IDH-", "p53+", "p53-", "ATRX+", "ATRX-"]
    # # list_colum = ['grade', '1p19q', 'idh', "tert", 'mgmt', 'idh.1', 'p53', 'atrx']

    # zero_arr = np.zeros((182, 16),dtype=float)
    # # zero_arr = np.zeros((8, 182),dtype=float)
    # data_heatmap = pd.DataFrame(zero_arr, columns=list_colum, index=list_row)
    # print("heatmap.head", data_heatmap.shape, data_heatmap.head(5))
    # data_heatmap.to_excel("heatmap.xlsx")

    #-------------------------------------------------------------------------------------------------
    # 对bingli和jiyin各运行一次
    # path3 = 'jiyin_feature_set_n.xlsx'
    # path3 = 'bingli_feature_set_n.xlsx'
    # df_new3 = pd.DataFrame()
    # df_new3.to_excel(path3, index=False)
    # writer3 = pd.ExcelWriter(path3, mode="a", engine="openpyxl")

    # data_heatmap = pd.read_excel("heatmap.xlsx", index_col=0)
    # # print("heatmap.head", data_heatmap.shape, data_heatmap.head(5))
    # # jiyin
    # # label_list =['grade', '1p19q', 'idh', 'tert', 'mgmt']
    # # bingli
    # label_list =['idh', 'p53', 'atrx']
    # # 定义大dataframe的value矩阵
    # # 对于每个分类，得到一个colum=[label,feature_set]， index = [training]的dataframe
    # sheet = "T1+T2+T1C+ADC+ASL+SWI"

    # for label in label_list:
    #     df1 = pd.read_excel("bingli_"+label+".xlsx", sheet_name=sheet)
    #     print("df1.shape_before",df1.shape)

    #     list1 = dfToList(path2, label)

    #     list1.append(label)
    #     list1.append('file name')

    #     df1 = df1[list1]
    #     print("df1.shape",df1.shape)

    #     # 特征标准化
    #     label_arr = np.array(df1[label].values.tolist())
    #     df1.drop(['file name', label], axis=1, inplace=True)
    #     feature_arr = df1.values  # 特征数组
    #     name_list = [label]
    #     for index, row in df1.iteritems():  # 多个字符串转array，先转list再转array
    #         name_list.append(index)
    #     name_arr = np.array(name_list)  # 特征名数组

    #     X_train = feature_arr
    #     y_train = label_arr
    #     # 标准化： StandardScaler
    #     scaler = StandardScaler()
    #     feature_train = scaler.fit_transform(X_train)

    #     # 将归一化或标准化的特征分别写到'jiyin/bingli_feature_set_n.xlsx'中
    #     # 四舍五入
    #     n_train = np.size(y_train)
    #     arr_train = np.append(y_train.reshape(n_train, 1), feature_train, axis=1)
    #     df_train2 = pd.DataFrame(arr_train, columns=name_arr)
    #     df_train2.to_excel(writer3, sheet_name = label, index = False)

    #     # 将dataframe按照labe分为2部分，记作两个dataframe，定义labe+和labe-
    #     df_positive = df_train2.loc[df_train2[label] == 1]
    #     df_negative = df_train2.loc[df_train2[label] == 0]
    #     df_positive.drop([label], axis=1, inplace=True)
    #     df_negative.drop([label], axis=1, inplace=True)

    #     # 将dataframe+和dataframe-的值(平均值)填到大dataframe中
    #     # 遍历所有列，求出均值
    #     if label == 'idh':
    #         label_p = label + '+.1'
    #         label_n = label + '-.1'
    #     else:
    #         label_p = label + '+'
    #         label_n = label + '-'

    #     # print("label", label_p)
    #     for index, column in df_positive.iteritems():
    #         col_mean = np.mean(column)
    #         data_heatmap.loc[index, label_p] = col_mean
    #         # print("type of loc return", type(data_heatmap.loc[index, label_p]))
    #     for index, column in df_negative.iteritems():
    #         col_mean = np.mean(column)
    #         # print("mean", col_mean)
    #         data_heatmap.loc[index, label_n] = col_mean
    #     # print("for complete!", data_heatmap)
        
    #     print("label complete!", label)
    # data_heatmap.to_excel("heatmap.xlsx")
    # writer3.save()
    # writer3.close()
            

    #---------------------------------------------------------------------------------------
    # 根据得到的xlsx画特征热力图
    data_important = pd.read_excel("features_important.xlsx", index_col=0)
    data_heatmap = pd.read_excel("heatmap.xlsx", index_col=0)

    # sns.set(font_scale=1.5)
    # plt.rc('font',family='Times New Roman',size=12)

    f, axes = plt.subplots(nrows=1, ncols=2, figsize = (8, 8))
    plt.subplots_adjust(wspace =0.08, hspace =0.08)
    axes = axes.flatten()

    #设置heatmap样式
    sns.set_style('dark')   
    cmap = sns.diverging_palette(220, 10, as_cmap=True)   # 设置颜色
    # print("cmap", cmap)

    cbar_ax = f.add_axes([.15, .03, .3, .03])

    ax0 = sns.heatmap(data_heatmap, cmap=cmap, square=True, ax = axes[0], cbar_ax=cbar_ax, cbar_kws={"orientation":"horizontal", "shrink": 0.5},  vmin=-0.7, vmax=0.6)

    ax0.set_title('Mean Feature value normalized', pad = 27.0)

    # x = [1, 2, 3]
    # y1 = np.array([2, 3, 2])
    # y2 = np.array([3, 1, 5])
    # y3 = np.array([1, 2, 1])
    # plt.bar(x, y1, color='green', label='y1')
    # plt.bar(x, y2, bottom=y1, color='red', label='y2')
    # plt.bar(x, y3, bottom=y1+y2, color='blue', label='y3')
    # plt.legend(loc=[1, 0])

    grade = data_important['Grade']
    pq = data_important['1p19q']
    idh = data_important['IDH']
    tert = data_important['TERT']
    mgmt = data_important['MGMT']
    idh1 = data_important['IDH.1']
    p53 = data_important['p53']
    atrx = data_important['ATRX']
    # print("atrx", atrx)

    #将数值映射为颜色，找出cmap所有颜色的名字
    counts = np.arange(0, 8, 1)
    norm = mpl.colors.Normalize(vmin=np.min(counts), vmax=np.max(counts))
    cmap = mpl.cm.get_cmap(cmap)
    color_list =  [cmap(norm(val)) for val in counts]
    # print("color_list",len(color_list), color_list)


    axes[1].barh(data_important.index, grade, alpha=0.6, color = color_list[0], label = 'Grade', align = 'edge')   
    axes[1].barh(data_important.index, pq, alpha=0.6, color = color_list[1], label = '1p19q', left = grade, align = 'edge')
    axes[1].barh(data_important.index, idh, alpha=0.6, color = color_list[2], label = 'IDH', left = grade + pq, align = 'edge')
    axes[1].barh(data_important.index, tert, alpha=0.6, color = color_list[3], label = 'TERT', left = grade + pq + idh, align = 'edge')
    axes[1].barh(data_important.index, mgmt, alpha=0.6, color = color_list[4], label = 'MGMT', left = grade + pq + idh + tert, align = 'edge')
    axes[1].barh(data_important.index, idh1, alpha=0.6, color = color_list[5], label = 'IDH.1', left = grade + pq + idh + tert + mgmt, align = 'edge')
    axes[1].barh(data_important.index, p53, alpha=0.6, color = color_list[6], label = 'p53', left = grade + pq + idh + tert + mgmt + idh1, align = 'edge')
    axes[1].barh(data_important.index, atrx, alpha=0.6, color = color_list[7], label = 'ATRX', left = grade + pq + idh + tert + mgmt + idh1 + p53, align = 'edge')
    axes[1].legend(loc = 'best')

    axes[1].spines['left'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['bottom'].set_visible(False)


    axes[1].set_yticks(())
    axes[1].grid(b=True, which='both', axis='x')

    axes[1].set_title('Overall Feature Importance')
    
    # ax.set_xlabel('region')
    # ax.set_ylabel('kind')

    f.savefig('heatmap_final.tif', dpi=1200, bbox_inches='tight')
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=-90)








