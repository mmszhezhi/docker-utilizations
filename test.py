from dockerutilizations.model.models import *
offset = 15000
train_data_normalized = scaler.fit_transform(df.values[15000:, :])

# train_data_normalized = df.values[:offset,:]
model = LSTM()
# model.load_state_dict(torch.load("D:\d\pytorch-repo\seqencelstm\model1"))
model.load_state_dict(torch.load("model1"))
model.eval()
loss_function = nn.MSELoss()
bsize,step,features = 99,150,2
seqs,targets = create_inout_sequences(train_data_normalized, step)
gen = genbatch(seqs,targets,500)
loss = 0
for i,(x, y) in enumerate(gen):
    with torch.no_grad():
        predict = model(x)
        pre = scaler.inverse_transform(predict)
        print(pre)
        avloss = loss_function(predict.squeeze(), y.squeeze())
        # truey = scaler.inverse_transform(y.squeeze())
        # predicty = scaler.inverse_transform(result.squeeze())
        loss += avloss
        print(f"batch {i} average loss:",avloss)
print("total loss:",loss)
