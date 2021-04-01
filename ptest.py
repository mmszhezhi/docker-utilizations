import requests
import json
import time
import pandas as pd
import random

df = pd.read_csv("data.csv")
start = random.randint(0,58456-250)

data=df.loc[start:start+149,['cpu','mem']]

params={
    "data": data.values.tolist(),
}

url='http://else.so:8079/predict'

time1=time.time()
html = requests.post(url, json.dumps(params))
print('发送post数据请求成功!')
print('返回post结果如下：')
print(html.text)

time2=time.time()
print('总共耗时：' + str(time2 - time1) + 's')
