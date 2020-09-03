import torch
import numpy as np

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


class MyReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx,input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod 
    def backward(ctx,grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input<0] =0
        return grad_input


x = torch.randn(300,30)
y = torch.randn(300,30)
model = TwoLayer(30,1000,30)
loss_fn = torch.nn.MSELoss(reduction='sum')
optim = torch.optim.Adam(model.parameters(), lr = 0.001)

for i in range(1000):
    pred = model(x)
    loss = loss_fn(pred,y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    if(i%200==0):
        print(pred,y)



print(y)
print(model(x))



