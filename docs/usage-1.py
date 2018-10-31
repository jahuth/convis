import numpy as np
import matplotlib.pylab as plt
import convis
import torch
l_goal = convis.models.LN()
k_goal = np.random.randn(5,5,5)
l_goal.conv.set_weight(k_goal)
plt.plot(l_goal.conv.weight.data.cpu().numpy()[0,0,:,:,:].mean(1))
plt.matshow(l_goal.conv.weight.data.cpu().numpy().mean((0,1,2)))
plt.colorbar()
l = convis.models.LN()
l.conv.set_weight(np.ones((5,5,5)),normalize=True)
l.set_optimizer.LBFGS()
# optional conversion to GPU objects:
#l.cuda()
#l_goal.cuda()
inp = 1.0*(np.random.randn(200,10,10))
inp = torch.autograd.Variable(torch.Tensor(inp)) # .cuda() # optional: conversion to GPU object
outp = l_goal(inp[None,None,:,:,:])
plt.figure()
plt.plot(l_goal.conv.weight.data.cpu().numpy()[0,0,:,:,:].mean(1),'--',color='red')
for i in range(50):
    l.optimize(inp[None,None,:,:,:],outp)
    if i%10 == 2:
        plt.plot(l.conv.weight.data.cpu().numpy()[0,0,:,:,:].mean(1))
plt.matshow(l.conv.weight.data.cpu().numpy().mean((0,1,2)))
plt.colorbar()
plt.figure()
h = plt.hist((l.conv.weight-l_goal.conv.weight).data.cpu().numpy().flatten(),bins=15)