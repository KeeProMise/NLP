import  torch
import torch.nn.functional as F
import numpy
# test = torch.rand(2,4,7)
#
# print(test.size())
# test = torch.cat((test[-2,:,:], test[-1,:,:]), dim = 1)
#
# print(test.size())
out = torch.rand(5,4,14)
out = out.permute([1,0,2])
# print(out.size())
test = torch.rand(4,6)
test = test.unsqueeze(1)
# print(test.size())
test = test.repeat(1,5,1)
now = torch.cat((test,out),dim=2)
att = torch.rand(4,5,8)
attention = torch.sum(att, dim=2)
attention = F.softmax(attention, dim=1)
attention = attention.unsqueeze(1)
weight = torch.bmm(attention,out)
print(weight.size())
