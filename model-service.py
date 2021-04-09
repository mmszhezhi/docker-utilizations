import uvicorn
from fastapi import FastAPI
import json
app = FastAPI()
import sys
sys.setrecursionlimit(sys.getrecursionlimit() * 5)
from model.models import *
offset = 15000
train_data_normalized = scaler.fit_transform(df.values[15000:, :])

# train_data_normalized = df.values[:offset,:]
model = LSTM()
model.load_state_dict(torch.load("model1"))
model.eval()
loss_function = nn.MSELoss()
bsize,step,features = 99,150,2

from pydantic import BaseModel

class Item(BaseModel):
    data: list
    description: str = None


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def read_item(item:Item):
    x = np.array(item.data).reshape([-1, 2])

    x = torch.FloatTensor(scaler.transform(x)[np.newaxis,:,:])

    with torch.no_grad():
        predict = model(x)
    pre = scaler.inverse_transform(predict)
    print(pre)
    return json.dumps({"cpu":pre[0][0],"mem":pre[0][1]})

if __name__ == '__main__':
    uvicorn.run(app, port=9081, host='0.0.0.0')