import torch
import numpy as np
from torchviz import make_dot
class TwoLayer(torch.nn.Module):
    
    def __init__(self,D_in,H,D_out):
        torch.nn.Module.__init__(self)
        self.input_layer = torch.nn.Linear(D_in,H)
        self.hidden_layer = torch.nn.Linear(H,H)
        self.hidden_layer1 = torch.nn.Linear(H,H)
        self.hidden_layer2 = torch.nn.Linear(H,H)
        self.output_layer = torch.nn.Linear(H,D_out)
    
    def forward(self,input):
        o = self.input_layer(input).clamp(min = 0)
        o = self.hidden_layer(o).clamp(min=0)
        o = self.hidden_layer1(o).clamp(min=0)
        o = self.hidden_layer2(o).clamp(min=0)
        pred = self.output_layer(o)
        return pred

class Test(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = TwoLayer(5,3,5)
        self.t2 = TwoLayer(5,3,5)
        self.partial = None

    def encode(self,x):
        self.partial = self.t1(x)
    
    def decode(self,y):
        a = self.t2(y)
        return a+self.partial

mod = Test()
q = torch.randn(10,5)
c = torch.randn(10,5)
mod.encode(q)
print("Encoded")
out = mod.decode(c)
j = torch.randn(10,5)
diff = out-j
print(diff)
loss = diff.sum()
# print(dir(mod.t1.))
# for p in mod.t1.hidden_layer.parameters():
#     print(dir(p))
#     print(p.grad)

    # print(p.weight.grad)
# print(mod.t1.hidden_layer.parameters())
# print(mod.t1.hidden_layer.parameters()[0].grad)
loss.backward()
for p in mod.t1.parameters():
    print(p,p.grad)
for p in mod.t2.hidden_layer.parameters():
    print(p.grad)
