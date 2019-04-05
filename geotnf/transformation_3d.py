from __future__ import print_function, division
import os
import sys
from skimage import io
import pandas as pd
import numpy as np
import torch
from torch.nn.modules.module import Module
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn.functional as F

class GeometricTnf(object):
    """
    
    Geometric transfromation to an image batch (wrapped in a PyTorch Variable)
    ( can be used with no transformation to perform bilinear resizing )        
    
    """
    def __init__(self, geometric_model='affine', out_h=240, out_w=240, use_cuda=True):
        self.out_h = out_h
        self.out_w = out_w
        self.use_cuda = use_cuda
        if geometric_model=='affine':
            self.gridGen = AffineGridGen(out_h, out_w)
        elif geometric_model=='tps':
            # Generate grid
            self.gridGen = TpsGridGen(out_h, out_w, use_cuda=use_cuda)
           
            
            
            
        self.theta_identity = torch.Tensor(np.expand_dims(np.array([[1,0,0],[0,1,0]]),0).astype(np.float32))
        if use_cuda:
            self.theta_identity = self.theta_identity.cuda()

    def __call__(self, image_batch, theta_batch=None, padding_factor=1.0, crop_factor=1.0):
        b, c, h, w = image_batch.size()
        if theta_batch is None:
            theta_batch = self.theta_identity
            theta_batch = theta_batch.expand(b,2,3)
            theta_batch = Variable(theta_batch,requires_grad=False)        
            
        sampling_grid = self.gridGen(theta_batch)

        # rescale grid according to crop_factor and padding_factor
        sampling_grid.data = sampling_grid.data*padding_factor*crop_factor
        # sample transformed image
        
        
        warped_image_batch = F.grid_sample(image_batch, sampling_grid)
        
        return warped_image_batch
    

class SynthPairTnf(object):
    """
    
    Generate a synthetically warped training pair using an affine transformation.
    
    """
    def __init__(self, use_cuda=True, geometric_model='affine', crop_factor=1.0, output_size=(240,240), padding_factor = 0.5):
        assert isinstance(use_cuda, (bool))
        assert isinstance(crop_factor, (float))
        assert isinstance(output_size, (tuple))
        assert isinstance(padding_factor, (float))
        self.use_cuda=use_cuda
        self.crop_factor = crop_factor
        self.padding_factor = padding_factor
        self.out_h, self.out_w = output_size 
        self.rescalingTnf = GeometricTnf('affine', self.out_h, self.out_w, 
                                         use_cuda = self.use_cuda)
        
        
        self.geometricTnf = GeometricTnf(geometric_model, self.out_h, self.out_w, 
                                         use_cuda = self.use_cuda)

        
    def __call__(self, batch):
        image_batch, theta_batch = batch['image'], batch['theta'] 
        if self.use_cuda:
            image_batch = image_batch.cuda()
            theta_batch = theta_batch.cuda()
            
        b, c, h, w = image_batch.size()
              
        # generate symmetrically padded image for bigger sampling region
        image_batch = self.symmetricImagePad(image_batch,self.padding_factor)
        
        # convert to variables
        image_batch = Variable(image_batch,requires_grad=False)
        theta_batch =  Variable(theta_batch,requires_grad=False)        

        # get cropped image
        cropped_image_batch = self.rescalingTnf(image_batch,None,self.padding_factor,self.crop_factor) # Identity is used as no theta given
        # get transformed image
        warped_image_batch = self.geometricTnf(image_batch,theta_batch,
                                               self.padding_factor,self.crop_factor) # Identity is used as no theta given
        
        return {'source_image': cropped_image_batch, 'target_image': warped_image_batch, 'theta_GT': theta_batch}

    def symmetricImagePad(self,image_batch, padding_factor):
        b, c, h, w = image_batch.size()
        pad_h, pad_w = int(h*padding_factor), int(w*padding_factor)
        idx_pad_left = torch.LongTensor(range(pad_w-1,-1,-1))
        idx_pad_right = torch.LongTensor(range(w-1,w-pad_w-1,-1))
        idx_pad_top = torch.LongTensor(range(pad_h-1,-1,-1))
        idx_pad_bottom = torch.LongTensor(range(h-1,h-pad_h-1,-1))
        if self.use_cuda:
                idx_pad_left = idx_pad_left.cuda()
                idx_pad_right = idx_pad_right.cuda()
                idx_pad_top = idx_pad_top.cuda()
                idx_pad_bottom = idx_pad_bottom.cuda()
        image_batch = torch.cat((image_batch.index_select(3,idx_pad_left),image_batch,
                                 image_batch.index_select(3,idx_pad_right)),3)
        image_batch = torch.cat((image_batch.index_select(2,idx_pad_top),image_batch,
                                 image_batch.index_select(2,idx_pad_bottom)),2)
        return image_batch

    
class AffineGridGen(Module):
    def __init__(self, out_h=240, out_w=240, out_ch = 3):
        super(AffineGridGen, self).__init__()        
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch
        
    def forward(self, theta):
        theta = theta.contiguous()
        batch_size = theta.size()[0]
        out_size = torch.Size((batch_size,self.out_ch,self.out_h,self.out_w))
        return F.affine_grid(theta, out_size)
        
class TpsGridGen(Module):
    def __init__(self, out_h=240, out_w=240, out_d=240, use_regular_grid=True, grid_size=3, reg_factor=0, use_cuda=True):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w, self.out_d = out_h, out_w, out_d
        
        self.reg_factor = reg_factor
        self.use_cuda = use_cuda

        # create grid in numpy
        self.grid = np.zeros( [self.out_h, self.out_w, self.out_d, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        
        self.grid_X, self.grid_Y, self.grid_Z = np.meshgrid(np.linspace(-1,1,out_w),np.linspace(-1,1,out_h),np.linspace(-1,1,out_d))
        
        # grid_X,grid_Y, grid_Z: size [1,H,W,D,1,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.grid_Z = torch.FloatTensor(self.grid_Z).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
 
        self.grid_X = Variable(self.grid_X,requires_grad=False)
        self.grid_Y = Variable(self.grid_Y,requires_grad=False)
        self.grid_Z = Variable(self.grid_Z,requires_grad=False)
        
        
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()
            self.grid_Z = self.grid_Z.cuda()

        # initialize regular grid for control points P_i
        if use_regular_grid:
            
            axis_coords = np.linspace(-1,1,grid_size)
            
            self.N = grid_size*grid_size*grid_size
            
            P_Y,P_X,P_Z = np.meshgrid(axis_coords,axis_coords,axis_coords)
            
            
            P_X = np.reshape(P_X,(-1,1)) # size (N,1)
            P_Y = np.reshape(P_Y,(-1,1)) # size (N,1)
            P_Z = np.reshape(P_Z,(-1,1)) # size (N,1)
            
            
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
            P_Z = torch.FloatTensor(P_Z)
            
            
            self.Li = Variable(self.compute_L_inverse(P_X,P_Y,P_Z).unsqueeze(0),requires_grad=False)
            
            
            self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4)
            self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4)
            self.P_Z = P_Z.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4)
            
            self.P_X = Variable(self.P_X,requires_grad=False)
            self.P_Y = Variable(self.P_Y,requires_grad=False)
            self.P_Z = Variable(self.P_Z,requires_grad=False)
            

            if use_cuda:
                self.P_X = self.P_X.cuda()
                
                self.P_Y = self.P_Y.cuda()

                self.P_Z = self.P_Z.cuda()

            
    def forward(self, theta):
        
        warped_grid = self.apply_transformation(theta,torch.cat((self.grid_X,self.grid_Y,self.grid_Z),4))
        
        return warped_grid
    
    def compute_L_inverse(self,X,Y,ZZ):
        N = X.size()[0] # num of points (along dim 0)
        # construct matrix K
        Xmat = X.expand(N,N)
        Ymat = Y.expand(N,N)
        Zmat = ZZ.expand(N,N)
        
        P_dist_squared = torch.pow(Xmat-Xmat.transpose(0,1),2)+torch.pow(Ymat-Ymat.transpose(0,1),2)+torch.pow(Zmat-Zmat.transpose(0,1),2)
        
        P_dist_squared[P_dist_squared==0]=1 # make diagonal 1 to avoid NaN in log computation
        K = torch.mul(P_dist_squared,torch.log(P_dist_squared))
        # construct matrix L
        O = torch.FloatTensor(N,1).fill_(1)
        Z = torch.FloatTensor(4,4).fill_(0)       
        P = torch.cat((O,X,Y,ZZ),1)
        L = torch.cat((torch.cat((K,P),1),torch.cat((P.transpose(0,1),Z),1)),0)
        Li = torch.inverse(L)
        if self.use_cuda:
            Li = Li.cuda()
        return Li
        
    def apply_transformation(self,theta,points):
        if theta.dim()==2:
            theta = theta.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        # points should be in the [B,H,W,D,3] format,
        # where points[:,:,:,:,0] are the X coords  
        # and points[:,:,:,:,1] are the Y coords  
        # and points[:,:,:,:,2] are the Z coords 
        
        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        Q_X=theta[:,:self.N,:,:,:].squeeze(4).squeeze(3).cuda()
        Q_Y=theta[:,self.N:self.N*2,:,:,:].squeeze(4).squeeze(3).cuda()
        Q_Z=theta[:,self.N*2:,:,:,:].squeeze(4).squeeze(3).cuda()

        
        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]
        points_d = points.size()[3]
       
        # repeat pre-defined control points along spatial dimensions of points to be transformed
        P_X = self.P_X.expand((1,points_h,points_w,points_d,1,self.N))
        P_Y = self.P_Y.expand((1,points_h,points_w,points_d,1,self.N))
        P_Z = self.P_Z.expand((1,points_h,points_w,points_d,1,self.N))
        
        
        # compute weigths for non-linear part
        W_X = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_X)
        W_Y = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_Y)
        W_Z = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_Z)

        
     
        # reshape
        # W_X,W,Y,W_Z: size [B,H,W,D,1,N]
        W_X = W_X.unsqueeze(3).unsqueeze(4).unsqueeze(5).transpose(1,5).repeat(1,points_h,points_w,points_d,1,1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).unsqueeze(5).transpose(1,5).repeat(1,points_h,points_w,points_d,1,1)
        W_Z = W_Z.unsqueeze(3).unsqueeze(4).unsqueeze(5).transpose(1,5).repeat(1,points_h,points_w,points_d,1,1)
#              # reshape
#         # W_X,W,Y: size [B,H,W,1,N]
            # B N 1 -- B N 1 1 1---B 1 1 1 N -- B H W 1 N
#         W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
#         W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
          
        # compute weights for affine part
        A_X = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,4,self.N)),Q_X)
        A_Y = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,4,self.N)),Q_Y)
        A_Z = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,4,self.N)),Q_Z)

        
        
        
        # reshape
        # A_X,A,Y: size [B,H,W,1,3]
        A_X = A_X.unsqueeze(3).unsqueeze(4).unsqueeze(5).transpose(1,5).repeat(1,points_h,points_w,points_d,1,1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).unsqueeze(5).transpose(1,5).repeat(1,points_h,points_w,points_d,1,1)
        A_Z = A_Z.unsqueeze(3).unsqueeze(4).unsqueeze(5).transpose(1,5).repeat(1,points_h,points_w,points_d,1,1)

        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        points_X_for_summation = points[:,:,:,:,0].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,:,0].size()+(1,self.N))
        points_Y_for_summation = points[:,:,:,:,1].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,:,1].size()+(1,self.N))
        points_Z_for_summation = points[:,:,:,:,2].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,:,2].size()+(1,self.N))
        
        if points_b==1:
            delta_X = points_X_for_summation-P_X
            delta_Y = points_Y_for_summation-P_Y
            delta_Z = points_Z_for_summation-P_Z
            
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation.cuda()-P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation.cuda()-P_Y.expand_as(points_Y_for_summation)
            delta_Z = points_Z_for_summation.cuda()-P_Z.expand_as(points_Z_for_summation)
            
        dist_squared = torch.pow(delta_X,2)+torch.pow(delta_Y,2)+torch.pow(delta_Z,2)
        # U: size [1,H,W,1,N]
        
        dist_squared[dist_squared==0]=1 # avoid NaN in log computation
        U = torch.mul(dist_squared,torch.log(dist_squared)) 
        
        # expand grid in batch dimension if necessary
        points_X_batch = points[:,:,:,:,0].unsqueeze(4)
        points_Y_batch = points[:,:,:,:,1].unsqueeze(4)
        points_Z_batch = points[:,:,:,:,2].unsqueeze(4)
        
        if points_b==1:
            points_X_batch = points_X_batch.expand((batch_size,)+points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,)+points_Y_batch.size()[1:])
            points_Z_batch = points_Z_batch.expand((batch_size,)+points_Z_batch.size()[1:])
       
    
        points_X_prime = A_X[:,:,:,:,:,0]+ \
                       torch.mul(A_X[:,:,:,:,:,1],points_X_batch.cuda()) + \
                       torch.mul(A_X[:,:,:,:,:,2],points_Y_batch.cuda()) + \
                       torch.mul(A_X[:,:,:,:,:,3],points_Z_batch.cuda()) + \
                       torch.sum(torch.mul(W_X,U.expand_as(W_X)),5)
                    
        points_Y_prime = A_Y[:,:,:,:,:,0]+ \
                       torch.mul(A_Y[:,:,:,:,:,1],points_X_batch.cuda()) + \
                       torch.mul(A_Y[:,:,:,:,:,2],points_Y_batch.cuda()) + \
                       torch.mul(A_Y[:,:,:,:,:,3],points_Z_batch.cuda()) + \
                       torch.sum(torch.mul(W_Y,U.expand_as(W_Y)),5)
        
        points_Z_prime = A_Z[:,:,:,:,:,0]+ \
                       torch.mul(A_Z[:,:,:,:,:,1],points_X_batch.cuda()) + \
                       torch.mul(A_Z[:,:,:,:,:,2],points_Y_batch.cuda()) + \
                       torch.mul(A_Z[:,:,:,:,:,3],points_Z_batch.cuda()) + \
                       torch.sum(torch.mul(W_Z,U.expand_as(W_Z)),5)
        
        #print(points_Z_prime.shape)

        fuck=torch.cat([points_X_prime,points_Y_prime,points_Z_prime],4)
        
        return fuck
        