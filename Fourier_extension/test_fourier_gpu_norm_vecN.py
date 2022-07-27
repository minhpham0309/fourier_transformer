import time
import math
import torch
import numpy as np
import fourier_N_cuda
import torch.nn as nn
#from GPUtil import showUtilization as gpu_usage

if torch.cuda.is_available():
    dev = torch.device('cuda')
    print('has cuda')
else:
    dev = "cpu"
    print('no cuda')
cuda_device = torch.device(dev)


class FOURIERFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, head_q, head_k, paramR):
        output = fourier_N_cuda.forward(head_q, head_k, paramR)
        variables = head_q, head_k, paramR, output
        ctx.save_for_backward(*variables)

        return output

    @staticmethod
    def backward(ctx, grad):
        head_q, head_k, paramR, output = ctx.saved_tensors
        grads = fourier_N_cuda.backward(
            grad.contiguous(), head_q, head_k, paramR, output)
        grad_head_q, grad_head_k, grad_p = grads
        return grad_head_q, grad_head_k, grad_p


class FourierAttn1(nn.Module):
    def __init__(self, n_head, d_model, d_head):
        super(FourierAttn1, self).__init__()

        self.n_head  = n_head
        self.d_model = d_model
        self.d_head  = d_head
        
        #torch.cuda.manual_seed(1);
        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False, device=cuda_device)
        self.k_net = nn.Linear(d_model, n_head * d_head, bias=False, device=cuda_device)
        
        torch.manual_seed(9)
        self.paramR = torch.randn( (n_head),  device = cuda_device, dtype=torch.float,  requires_grad=True)

    def forward(self, h, mems):
        # h: size = [qlen, bsz, d_model] = [256, 48, 128]
        # q_net: shape = [d_model, n_head*d_head]  = [128,8x8=64]
        
        c = torch.cat( [mems,h], 0)   #size = [klen, bsz, d_model]

        head_q = self.q_net(h) # shape: [256, 48, 128] = [qlen, bsz, n_head*d_head]
        head_k = self.k_net(c) # shape: [512, 48, 128] = [klen, bsz, n_head*d_head]

        # [hlen, bsz, n_head, d_head]
        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head) 
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
                
        output = FOURIERFunction.apply( head_q,head_k, self.paramR)
        output = torch.abs(output);
            
        return output
        
        
        
class FourierAttn2(nn.Module):
    def __init__(self, n_head, d_model, d_head):
        super(FourierAttn2, self).__init__()

        self.n_head  = n_head
        self.d_model = d_model
        self.d_head  = d_head

        #torch.cuda.manual_seed(1);
        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False, device=cuda_device)
        self.k_net = nn.Linear(d_model, n_head * d_head, bias=False, device=cuda_device)

        torch.manual_seed(9)
        self.paramR = torch.randn( (n_head),  device = cuda_device, dtype=torch.float,  requires_grad=True)

    def forward(self, h, mems):
        # h: size = [qlen, bsz, d_model] = [256, 48, 128]
        # q_net: shape = [d_model, n_head*d_head]  = [128,8x8=64]
        
        c = torch.cat( [mems,h], 0)   #size = [klen, bsz, d_model]

        head_q = self.q_net(h) # shape: [256, 48, 128] = [qlen, bsz, n_head*d_head]
        head_k = self.k_net(c) # shape: [512, 48, 128] = [klen, bsz, n_head*d_head]
        #head_v = self.v_net(c)

        # [hlen, bsz, n_head, d_head]
        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head) 
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        
        ########################################
        # [n_head, bsz, qlen, d_head]
        head_q = head_q.transpose(0,2)
        head_k = head_k.transpose(0,2)
        
        # [n_head, bsz, qlen, klen, d_head]
        QK_distance0 = (head_q.unsqueeze(3) - head_k.unsqueeze(2)) * self.paramR[:,None,None,None,None] / math.pi 
        QK_distance0 = torch.sinc(QK_distance0) 
        output = torch.prod(QK_distance0, dim=4)  # [n_head, bsz, qlen, klen]

        output = torch.abs(output);
        output = output.permute(2, 3, 1, 0) # qlen x klen x bsz x n_head
            
        return output
        


bsz = 32
n_head = 8
d_head = 16
mlen = 96
qlen = 128      #256
klen = mlen + qlen
d_model = 128   #128
'''
[120,40,128] -> [120,40, 8*16] -> [120,40,8,16] -> [8,40,120,16]
'''

torch.manual_seed(5)
C1 = torch.randn( (qlen, bsz, d_model), device = 0, dtype=torch.float, requires_grad=True)
torch.manual_seed(5)
C2 = torch.randn( (qlen, bsz, d_model), device = 0, dtype=torch.float, requires_grad=True)

torch.manual_seed(3)
mem1 = torch.randn( (mlen, bsz, d_model),  device = cuda_device, dtype=torch.float,  requires_grad=True)
torch.manual_seed(3)
mem2 = torch.randn( (mlen, bsz, d_model),  device = cuda_device, dtype=torch.float, requires_grad=True)

Y = torch.randn( (qlen, klen, bsz, n_head), dtype=torch.float, generator=torch.Generator().manual_seed(1), requires_grad=True).cuda()


torch.manual_seed(1)
model1 = FourierAttn1(n_head, d_model, d_head)

torch.manual_seed(1)
model2 = FourierAttn2(n_head, d_model, d_head)

loss_fn = torch.nn.MSELoss(reduction='sum')



# model 1
time1=0;
start=time.time() 
for i in range(100):
  Y1 = model1(C1,mem1)
  loss1 = loss_fn(Y1,Y)
  model1.zero_grad()
  loss1.backward();
  #print()
time1 +=  (time.time()-start);


time2=0;
start=time.time() 
for i in range(100):
  Y2 = model2(C2,mem2)
  loss2 = loss_fn(Y2,Y)
  model2.zero_grad()
  loss2.backward(retain_graph=True);
time2 +=  (time.time()-start);



print('\n',time1, ', ', time2 )
print('Y1: ',Y1[0,1,2,3:5], '\nY2: ',Y2[0,1,2,3:5], )
print('norm(Y1-Y2)=', torch.sum(torch.abs(Y1-Y2)), ', ', torch.sum(torch.abs(Y2)) )

print('model1.q_net.weight.grad:', model1.q_net.weight.grad[1,4:7])
print('model2.q_net.weight.grad:', model2.q_net.weight.grad[1,4:7])
print('norm(q1-q2)=', torch.sum(torch.abs(model1.q_net.weight.grad-model2.q_net.weight.grad)), ', ', torch.sum(torch.abs(model1.q_net.weight.grad)) )

print('model1.k_net.weight.grad:', model1.k_net.weight.grad[1,4:7])
print('model2.k_net.weight.grad:', model2.k_net.weight.grad[1,4:7])
print('norm(k1-k2)=', torch.sum(torch.abs(model1.k_net.weight.grad-model2.k_net.weight.grad)), ', ', torch.sum(torch.abs(model1.k_net.weight.grad)) )

print('model1.paramR.grad:', model1.paramR.grad[1:4] )
print('model2.paramR.grad:', model2.paramR.grad[1:4] )
print('norm(R1-R2)=', torch.sum(torch.abs(model1.paramR.grad-model2.paramR.grad)), ', ', torch.sum(torch.abs(model1.paramR.grad)) )

#print(C1[0,2,1], C2[0,2,1])
print('grad(C1):', C1.grad[0,2,3:6])
print('grad(C2):', C2.grad[0,2,3:6])

print('grad(mem1): ', mem1.grad[2,1,3:6])
print('grad(mem2): ', mem2.grad[2,1,3:6])


