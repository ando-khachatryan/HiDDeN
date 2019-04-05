import torch
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)
def repeat_message(message, mult):
    shape = message.shape
    message.unsqueeze_(1).unsqueeze_(1)
    message = message.expand(-1, mult, mult, -1, -1)
    message = message.permute(0, 1, 3, 2, 4)
    message = message.reshape(shape[0], mult*shape[1], mult*shape[2])
    return message

batch=2
t = torch.empty((2,8,8), dtype=torch.int)
print(f't.shape: {t.shape}')
for b in range(t.shape[0]):
    for i in range(t.shape[1]):
        for j in range(t.shape[2]):
            t[b,i,j]= 100*b + 10*i + j

rep = repeat_message(t, 3)
print(rep.shape)
print(rep.numpy())

for i in rep.shape:


# t.unsqueeze_(1).unsqueeze_(1)
# print(t.shape)
# mult=3
# t=t.expand(t.shape[0], mult, mult, -1, -1)
# print(t.shape)
# t = t.permute(1, 3, 2, 4)
# print(t.shape)
# t=t.reshape(24,24)
# print(t.shape)
# print(t)




