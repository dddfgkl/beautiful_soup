import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import pickle

dtype = torch.FloatTensor

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




input_batch = Variable(torch.LongTensor(inputs))
target_batch = Variable(torch.LongTensor(targets))


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
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

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(100):
    optimizer.zero_grad()
    output = model(input_batch)

    # output : [batch_size, num_classes], target_batch : [batch_size] (LongTensor, not one-hot)
    loss = criterion(output, target_batch)

    corrects = (torch.max(output, 1)[1].view(targets.size()).data == targets).sum()
    print(corrects)
    accuracy = corrects / targets.size()

    if (epoch + 1) % 2 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss), 'acc = ', accuracy)

    loss.backward()
    optimizer.step()

word2cnt = data["word2cnt"]
cnt2word = data["cnt2word"]
# Test
tests = data["data"][:10]

test_batch = Variable(torch.LongTensor(tests))

# Predict
test_result = model(test_batch).data
print("test_result:")
print(test_result)

predict = model(test_batch).data.max(1, keepdim=True)[1]

if predict[0][0] == 0:
    print(test_text,"is Bad Mean...")
else:
    print(test_text,"is Good Mean!!")