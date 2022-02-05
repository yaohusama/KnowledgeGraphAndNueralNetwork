import pickle
import numpy as np
import pandas

keywords=pickle.load(open("Source/keywords.pkl", "rb"))
num=pickle.load(open("Source/count_all.pkl", "rb"))
org=pickle.load(open("Source/org.pkl", "rb"))
label_list=pickle.load(open("Source/label.pkl", "rb"))
print(num)
res=[]
label=[]
for i,item in enumerate(zip(keywords,org,label_list)):
    if i<num:
        tmp=np.concatenate((item[0].reshape(-1,200),item[1].reshape(-1,200)),axis=1)
        res.append(np.squeeze(tmp,axis=0))
        # print("res",np.array(res).shape)
        label.append(item[2])
result={}
result["label"]=label
result["data"]=res
result1=pandas.DataFrame(result)
result1.to_csv("result.csv")
with open("Source/data.pkl","wb") as f1:
    pickle.dump(result,f1)