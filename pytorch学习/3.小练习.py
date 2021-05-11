# 拟合y = w1x²+w2x+b
# y = -x²+2x+3

import torch

x_data = [1.0, 2.0, 3.0]
y_data = [4.0, 3.0, 0.0]

w1 = torch.tensor([1.0], requires_grad = True)
w2 = torch.tensor([1.0], requires_grad = True)
b = torch.tensor([1.0], requires_grad = True)

def forward(x):
    return w1*x*x + w2*x + b

def loss(x, y):
    y_pred = forward(x)
    return (y - y_pred) ** 2

for epoch in range(500):
    print("++++++++++epoch",epoch+1,"++++++++++")
    for x, y in zip (x_data, y_data):
        l = loss(x, y)
        l.backward()
        w1.data -= 0.01 * w1.grad.data
        w2.data -= 0.01 * w2.grad.data
        b.data -= 0.01 * b.grad.data
        w1.grad.data.zero_()
        w1.grad.data.zero_()
        w1.grad.data.zero_()
    print("loss:",l.item(),"w1:",w1.item(),"w2:",w2.item(),"b:",b.item())

print(forward(1.0))
print(forward(2.0))
print(forward(3.0))