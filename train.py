import torch
import LoaderFish
import os
import sys
import numpy as np
import matplotlib.pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import geotnf.point_tnf
##########################################
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="0"
deformation_list=[0.5]
lr = 0.00001 # learning rate
EP=4 #epoch
train_size=20000 #size for training data
test_size=1000 # size for testing data
name="Vis_ts_{}_EP_{}_dl_{}".format(train_size,EP,deformation_list)  # to save result
###########################################

def chamfer_loss(x,y, ps=91):
    A=x.cuda()
    B=y.cuda()
    A=A.permute(0,2,1)
    B=B.permute(0,2,1)
    r=torch.sum(A*A,dim=2)
    r=r.unsqueeze(-1)
    r1=torch.sum(B*B,dim=2)
    r1=r1.unsqueeze(-1)
    t=(r.repeat(1,1,ps) -2*torch.bmm(A,B.permute(0,2,1)) + r1.permute(0, 2, 1).repeat(1,ps,1))
    d1,_=t.min(dim=1)
    d2,_=t.min(dim=2)
    ls=(d1+d2)/2
    return ls.mean()

def gaussian_mix_loss(x,y, var=1,ps=91,w=0, sigma=20):
    #center is B
    A=x.cuda()
    B=y.cuda()
    A=A.permute(0,2,1)
    B=B.permute(0,2,1)
    bs=A.shape[0]
    ps=A.shape[1]
    A=(A.unsqueeze(2)).repeat(1,1,ps,1)
    B=(B.unsqueeze(1)).repeat(1,ps,1,1)
    sigma_inverse=((torch.eye(2)*(1.0/var)).unsqueeze(0).unsqueeze(0).unsqueeze(0)).repeat([bs, ps,ps,1,1]).cuda()
    sigma_inverse=sigma*sigma_inverse
    sigma_inverse=sigma_inverse.view(-1,2,2)
    tmp1=(A-B).unsqueeze(-2).view(-1,1,2)      
    tmp=torch.bmm(tmp1, sigma_inverse)
    tmp=torch.bmm(tmp,tmp1.permute(0,2,1))
    tmp=tmp.view(bs,ps,ps)
    tmp=torch.exp(-0.5*tmp)
    tmp=tmp/(2*np.pi*var)
    tmp1=tmp.sum(dim=-1)
    tmp2=tmp.sum(dim=1)
    tmp1=torch.clamp(tmp1, min=0.01)  
    return (-torch.log((tmp1)/90.0)).mean()

def cd_batch(aa, aa1, bb1, cname):
    j=0
    B=[]
    A=[]
    for j in range(len(aa)):
        d1=np.asarray((aa[j])).T  
        dd2=np.asarray((aa1[j])).T
        d2=np.asarray((bb1[j])).T
        aaa=chamfer_loss(torch.from_numpy(d1).unsqueeze(0).permute(0,2,1),torch.from_numpy(dd2).unsqueeze(0).permute(0,2,1),ps=x.shape[-1])
        aaa=aaa.cpu().numpy()
        A.append(aaa)       
        bbb=chamfer_loss(torch.from_numpy(d1).unsqueeze(0).permute(0,2,1),torch.from_numpy(d2).unsqueeze(0).permute(0,2,1),ps=x.shape[-1])
        bbb=bbb.cpu().numpy()
        B.append(bbb)
    return A,B


class DriftNet(nn.Module):
    def __init__(self, ks = 12, ffs=5, ps=91):
        super(DriftNet, self).__init__()
        self.ffs = ffs
        self.ks = ks **2
        self.kks=ks

        # creating ref points
        a=np.asarray([[-1.0+2.0/(ks-1)*i,1.0] for i in range(ks)])
        ref=[a-i*np.asarray([0,2.0/(ks-1)]) for i in range(ks)]
        a=np.asarray(ref).reshape(-1,2).astype(np.float32)
        self.ref=torch.from_numpy(a)
        self.ref = Variable(self.ref.cuda(), requires_grad=False)
        self.conv1 = nn.Conv1d(4,16,1)
        self.conv2 = nn.Conv1d(16,32,1)
        self.conv3 = nn.Conv1d(32,64,1)
        self.conv4 = nn.Conv1d(64,128,1)
        self.conv5 = nn.Conv1d(128,256,1)
        self.conv55 = nn.Conv1d(256,512,1)
        self.conv555 = nn.Conv1d(512,1024,1)
        self.mp1 = nn.MaxPool1d(ps)
        self.conv6 = nn.Conv2d(1,128,kernel_size=(3,3),stride=2)
        self.conv7 = nn.Conv2d(128,256,kernel_size=(4,4),stride=2)
        self.conv8 = nn.Conv2d(256,512,kernel_size=(5,5),stride=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn55 = nn.BatchNorm1d(512)
        self.bn555 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm2d(128)
        self.bn7 = nn.BatchNorm2d(256)
        self.bn8 = nn.BatchNorm2d(512)
        self.bn9 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(512,64)
        self.fc2 = nn.Linear(64,18)

    def forward(self, x, y):
        bs = x.size(0)
        ps = x.size(2)
        ref = self.ref.unsqueeze(0).unsqueeze(0).repeat(bs, ps, 1,1)
        x = x.permute(0,2,1).unsqueeze(2).repeat(1,1,self.ks,1).contiguous()
        x = torch.cat([x, ref],dim=-1)
        y = y.permute(0,2,1).unsqueeze(2).repeat(1,1,self.ks,1).contiguous()
        y = torch.cat([y, ref],dim=-1)
        x = x.contiguous().view(bs,-1,4).permute(0,2,1)
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = F.leaky_relu(self.bn5(self.conv5(x)))
        x = self.mp1(x)
        x = x.contiguous().view(bs, -1, self.ks)
        y = y.contiguous().view(bs,-1,4).permute(0,2,1)
        y = F.leaky_relu(self.bn1(self.conv1(y)))
        y = F.leaky_relu(self.bn2(self.conv2(y)))
        y = F.leaky_relu(self.bn3(self.conv3(y)))
        y = F.leaky_relu(self.bn4(self.conv4(y)))
        y = F.leaky_relu(self.bn5(self.conv5(y)))
        y = self.mp1(y)
        y = y.contiguous().view(bs, -1, self.ks).permute(0,2,1)
        inner = torch.bmm(y,x).view(bs, -1, self.ks).unsqueeze(1)
        inner = inner.contiguous()
        inner = F.leaky_relu(self.bn6(self.conv6(inner)))
        inner = F.leaky_relu(self.bn7(self.conv7(inner)))
        inner = F.leaky_relu(self.bn8(self.conv8(inner)))
        inner = F.max_pool2d(inner, kernel_size=inner.size()[2:])
        inner = inner.contiguous().view(bs, -1)
        out = F.leaky_relu(self.fc1(inner))
        out = self.fc2(out)
        theta = np.array([-1 , -1 , -1 , 0 , 0 , 0 , 1 , 1 , 1 , -1 , 0 , 1 , -1 , 0 , 1 , -1 , 0 , 1])
        theta = Variable(torch.from_numpy(theta).float()).cuda().unsqueeze(0).repeat(bs,1)
        return out+theta

CD_bef=[] # chamfer distance before registration
CD_aft=[] # chamfer distance after registration

for deformation in deformation_list:
    print(".......Synthesizing Training Pairs......")
    a=LoaderFish.PointRegDataset(total_data=train_size, 
                  deform_level=deformation,
                  noise_ratio=0, 
                  outlier_ratio=0, 
                  outlier_s=False,
                    outlier_t=True, 
                    noise_s=False, 
                    noise_t=True,
                  missing_points=0,
                  miss_source=False,
                    miss_targ=True)
    
    data_loader = torch.utils.data.DataLoader(a, batch_size=16, shuffle=True)
    
    print(".......Synthesizing Testing Pairs......")    
    b=LoaderFish.PointRegDataset(total_data=test_size, 
                  deform_level=deformation,
                  noise_ratio=0, 
                  outlier_ratio=0, 
                  outlier_s=False,
                    outlier_t=True, 
                    noise_s=False, 
                    noise_t=True,
                  missing_points=0,
                  miss_source=False,
                            miss_targ=True)
    
    data_loader_test = torch.utils.data.DataLoader(b, batch_size=10, shuffle=False)

    for batch_idx, batch in enumerate(data_loader):
        at=batch
        break
    #read a sample data
    x=at[0]
    y=at[1]
    theta=at[2]
        
    trs=geotnf.point_tnf.PointTnf(use_cuda=True)
    net = DriftNet(ps=x.shape[-1])
    net = net.cuda()
    print(net)
    
    solver = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
    op_schedule=optim.lr_scheduler.StepLR(solver,step_size=100, gamma=0.99)
    use_gpu = torch.cuda.is_available()
    net = nn.DataParallel(net)
    net.cuda()
    cudnn.benchmark = True  

    for epoch in range(0, EP):
        net.train()
        total_loss = 0.0
        op_schedule.step(epoch)
        for batch_idx, (x,y,theta,_) in enumerate(data_loader):
            print(batch_idx)
            x, y, theta = x.cuda(), y.cuda(), theta.cuda()        
            x, y, theta = Variable(x).float(), Variable(y).float(), Variable(theta).float()
            theta_hat=net(x,y)
            yhat=trs.tpsPointTnf(theta_hat,y)
            loss_cls = gaussian_mix_loss(x, yhat, ps=x.shape[-1],w=0, sigma=min(100,batch_idx//3+1))
            loss_cham=chamfer_loss(x,yhat,ps=x.shape[-1])
            solver.zero_grad()
            loss_cls.backward()
            solver.step()
#             if batch_idx%100==0:
#                 print(epoch)
#                 print(batch_idx, loss_cls.data.cpu().numpy())
    
    #save weights
    #torch.save(net.state_dict(), name+'.pth')
    
    AA=[]
    BB=[]
    
    for batch_idx, (x,y,theta,_) in enumerate(data_loader_test):
            x, y, theta = x.cuda(), y.cuda(), theta.cuda()        
            x, y, theta = Variable(x).float(), Variable(y).float(), Variable(theta).float()

            sc=y.cpu().numpy()
            d2=np.asarray((sc[0])).T
            c=d2[:,1]+d2[:,0]
            theta_hat=net(x,y)
            yhat=trs.tpsPointTnf(theta_hat,y)
            loss_cls = gaussian_mix_loss(x, yhat, ps=x.shape[-1],w=0, sigma=min(100,batch_idx//3+1))
            loss_cham=chamfer_loss(x,yhat,ps=x.shape[-1])
            a,b=cd_batch(x.cpu().numpy(),y.cpu().numpy(),yhat.data.cpu().numpy(),"Fish_batch_"+str(batch_idx))    
            print("before:",a)
            print("after",b)
            AA=AA+a
            BB=BB+b
            print("*************************************")
    
    #save for visulization
    from scipy.io import savemat
    savemat(name+"_visulization_"+"{}".format(deformation)+".mat", {"s":x.cpu().numpy(),"ts":yhat.data.cpu().numpy(), "t":y.cpu().numpy()})
    globals()["AA_{}".format(deformation).replace(".","_")]=AA
    globals()["BB_{}".format(deformation).replace(".","_")]=BB
    CD_bef.append([np.mean(AA),np.std(AA)])
    CD_aft.append([np.mean(BB),np.std(BB)])
    
np.savetxt(name,np.asarray(CD_bef+CD_aft))
