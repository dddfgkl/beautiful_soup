import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import pickle
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dtype = torch.FloatTensor
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


train_f = "/home/machong/workspace/data/classification/Chinese_conversation/h5train.pkl"
f1 = open(train_f, 'rb')
data = pickle.load(f1)
inputs = data["data"]
targets = data["label"]

# Text-CNN Parameter
embedding_size = 2 # n-gram
num_classes = 2  # 0 or 1
filter_sizes = [2, 2, 2] # n-gram window
num_filters = 3
vocab_size = len(data["word2cnt"])


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size).to(device)
        self.conv1ds = nn.ModuleList(
            [nn.Sequential(
                nn.Conv1d(embedding_size, num_filters, kernel_size=filter_sizes[index]),
                nn.ReLU()) for index in range(len(filter_sizes))]
        )
        self.fc1 = nn.Linear(len(filter_sizes)*num_filters, num_classes)

    def forward(self, x):
        out = self.embed(x)
        out = torch.transpose(out, 1, 2)
        out = [conv(out) for conv in self.conv1ds]
        out = [F.max_pool1d(c_out, kernel_size= c_out.size(2)).squeeze(2) for c_out in out]
        out = torch.cat(out, 1)
        logit = self.fc1(out)
        return logit

model = TextCNN()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(1000):
    input_batch = Variable(torch.LongTensor(inputs)).to(device)
    target_batch = Variable(torch.LongTensor(targets)).to(device)
    optimizer.zero_grad()
    output = model(input_batch)

    # output : [batch_size, num_classes], target_batch : [batch_size] (LongTensor, not one-hot)
    loss = criterion(output, target_batch)

    corrects = (torch.max(output, 1)[1].view(target_batch.size()).data == target_batch.data).sum()
    # print(corrects.item(), type(corrects.item()))
    accuracy = corrects.item() / len(targets)
    # print(accuracy)

    if (epoch + 1) % 2 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss), 'acc = ', accuracy)

    loss.backward()
    optimizer.step()

word2cnt = data["word2cnt"]
cnt2word = data["cnt2word"]
# Test
tests = data["data"][:100]

test_batch = Variable(torch.LongTensor(tests)).to(device)

# Predict
test_result = model(test_batch).data
test_gound_truth = data["label"][:100]
print("test_result:")
print(test_result)

predict = model(test_batch).data.max(1, keepdim=True)[1]

for i in range(len(tests)):
    sent = " ".join([cnt2word[word] for word in tests[i]])
    # print("### ", sent)
    if predict[i][0] != test_gound_truth[i]:
        print(sent)