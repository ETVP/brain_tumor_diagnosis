import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mpl


def count(path, str):
    print("str", str)
    data1 = pd.DataFrame(pd.read_excel(path, sheet_name = str))
    # 6模态的特征列表
    list2 = []
    list1 = data1.values.tolist()[3][0]
    list1 = list1.lstrip('[')
    list1 = list1.rstrip(']')
    list1 = list1.split(',')
    list1.remove(list1[0])
    for i in range(len(list1)):
        list2.append(eval(list1[i]))
    
    print("list2", list2)

    feature_len = len(list2)
    print("len", feature_len)

    # 定义统计变量的字典
    counts = {'Wavelet': 0, 'Original': 0, 'Clinic': 0, 'T1': 0, 'T2': 0, 
                'T1C': 0, 'ADC': 0, 'ASL': 0, 'SWI': 0, 'Shape': 0, 'First-order': 0, 'Texture': 0, 'Clinic2': 0}
    for i in range(feature_len):
        if 'wavelet' in list2[i]:
            counts['Wavelet'] += 1
        elif 'original' in list2[i]:
            counts['Original'] += 1
        else:
            counts['Clinic'] += 1
        
        if 'SWI' in list2[i]:
            counts['SWI'] += 1
        elif 'T2' in list2[i]:
            counts['T2'] += 1
        elif 'T1C' in list2[i]:
            counts['T1C'] += 1
        elif 'ADC' in list2[i]:
            counts['ADC'] += 1
        elif 'ASL' in list2[i]:
            counts['ASL'] += 1
        else:
            counts['T1'] += 1

        if 'shape' in list2[i]:
            counts['Shape'] += 1
        elif 'firstorder' in list2[i]:
            counts['First-order'] += 1
        elif ('glcm' in list2[i]) or ('glszm' in list2[i]) or ('gldm' in list2[i]) or ('glrlm' in list2[i]) or ('ngtdm' in list2):
            counts['Texture'] += 1
        else:
            counts['Clinic2'] += 1
    print("===============================counts", counts)


def survey(results, category_names, ax, flag):
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)

    sns.set_style('dark')   
    cmap = sns.diverging_palette(220, 10, as_cmap=True)   # 设置颜色
    category_colors = plt.get_cmap(cmap)(
        np.linspace(0.05, 0.85, data.shape[1]))

    
    ax.invert_yaxis()
    # ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    if flag != 1:
        ax.yaxis.set_visible(False)

    i = 0 
    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)
        
        # print(data_cum[:, i])
        # r, g, b, _ = color
        # text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        # ax.bar_label(rects, label_type='center', color=text_color)
    print("---------------------------")
    # ax.text(data_cum[:, i]+0.05,labels,'1',fontsize=14,horizontalalignment='center')
    ax.legend(loc = 'upper right', fontsize='small')
    ax.grid(b=True, which='both', axis='x')
    ax.set_xlim(0, 110)
    ax.set_xticks(np.arange(0, 110, 20))
    print(data_cum[:, i])

    if flag == 1:
        data_top = data[:,0]
        text_set = ((data_top/data_cum[:, i])*100).round(1)
        
    elif flag == 2:
        data_top = data[:,3] + data[:,4] + data[:,5]
        text_set = ((data_top/data_cum[:, i])*100).round(1)
    else:
        data_top = data[:,2]
        text_set = ((data_top/data_cum[:, i])*100).round(1)
    

    for x,y,z in zip(data_cum[:, i],labels, text_set):
        ax.text(x+0.5,y,str(z)+'%',fontsize=8,verticalalignment='center')


if __name__ == "__main__":
    # 统计次数 
    # jiyin
    # path = "C:/Users/89493/Desktop/final_outcom/jiyin_features.xlsx"
    # label_list =['grade', '1p19q', 'idh', 'tert', 'mgmt']
    # bingli          'idh', 
    # label_list = ['p53', 'atrx']
    # path = "C:/Users/89493/Desktop/final_outcom/bingli_features.xlsx"
    # for label in label_list:
    #     count(path, label)

    # 'Wavelet', 'Original', 'Clinic'

    category_names = ['Wavelet', 'Original', 'Clinic']
    results = {
        'Grade': [11, 2, 0],
        
        'IDH': [26, 8, 1],
        'TERT': [24, 7, 1],
        'MGMT': [18, 2, 0],
        '1p19q': [38, 7, 0],
        'IDH.1': [9, 1, 1],
        'p53':[7, 0, 0],
        'ATRX':[29, 3, 3]
    }

    # 'T1', 'T2', 'T1C', 'ADC', 'ASL', 'SWI'
    category_names1 = ['T1W', 'T2W', 'CE-T1W', 'ADC', 'ASL', 'SWI']
    results1 = {
        'grade': [1, 1, 5, 3, 0, 3],
        
        'idh': [4, 7, 11, 1, 8, 4],
        'tert': [7, 5, 3, 3, 3, 11],
        'mgmt': [2, 3, 6, 1, 5, 3],
        '1p19q': [6, 10, 9, 6, 8, 6],
        'idh.1': [2, 0, 4, 0, 2, 3],
        'p53':[3, 0, 1, 1, 1, 1],
        'atrx':[11, 5, 4, 5, 8, 2]
    }

    # 'Shape', 'First-order', 'Texture', 'Clinic'
    category_names2 = ['Shape', 'First-order', 'Texture', 'Clinic']
    results2 = {
        'grade': [0, 7, 6, 0],
        
        'idh': [0, 14, 20, 1],
        'tert': [0, 9, 21, 2],
        'mgmt': [0, 6, 14, 0],
        '1p19q': [0, 10, 35, 0],
        'idh.1': [0, 3, 7, 1],
        'p53':[0, 1, 6, 0],
        'atrx':[0, 10, 22, 3]
    }

    f, axes = plt.subplots(nrows=1, ncols=3, figsize = (8, 3))
    plt.subplots_adjust(wspace =0.08, hspace =0.08)
    ax = axes.flatten()
    
    survey(results, category_names, ax[0], 1)
    survey(results1, category_names1, ax[1], 2)
    survey(results2, category_names2, ax[2], 3)

    f.savefig('feature_bar_ra.tif', dpi=1500, bbox_inches='tight')
    plt.show()