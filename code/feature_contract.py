import os
from radiomics import featureextractor, getTestCase
import pandas as pd

paramDir = r'D:\Work\brain_tumor\feature'
params = os.path.join(paramDir, "Params.yaml") # 参数文件路径

# dataDir = r'D:\Work\brain_tumor\samples\103634\ADC\nii' # 存放有image和mask的文件夹
sampleDir = r'D:\Work\brain_tumor\samples'
samplesList = os.listdir(sampleDir)
# print(type(samplesList))

featureList = []          # 特征列表
nameList = []             #病历号列表

#遍历所有病例（文件夹）提取特征
for file in samplesList:
  # print(file)
  nameList.append(file)

  imageDir = os.path.join(sampleDir, file, "T1C")
  imageName = os.path.join(imageDir, "rbrain_image.nii")
  maskDir = os.path.join(sampleDir, file)
  maskName = os.path.join(maskDir, "3 OAx T2 frFSE-label.nii")

  extractor = featureextractor.RadiomicsFeatureExtractor(params) # 使用参数文件实例化特征提取器类
  extractor.addProvenance(False)     #不用显示附加信息
  # extractor.enableImageTypeByName('Wavelet')

  result = extractor.execute(imageName, maskName)#提取特征
  featureList.append(result)
  # print("dictionary len: ", len(result))
  # for key, val in result.items():
  #   print("\t", key, ":", val) # 进行特征提取

# for feature in featureList:
#   print("feature: ", feature)
# for name in nameList:
#   print("name: ", name)

#将nameList和featureList导出到excel表格中
df = pd.DataFrame(featureList, index = nameList)
df.to_excel('T1C_feature_extract.xlsx')



