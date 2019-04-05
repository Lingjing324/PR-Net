import torch
from torch.autograd import Variable
import numpy as np
from geotnf.transformation_3d import TpsGridGen

class PointTnf(object):
    """
    
    Class with functions for transforming a set of points with affine/tps transformations
    
    """
    def __init__(self, use_cuda=True):
        self.use_cuda=use_cuda
        self.tpsTnf = TpsGridGen(use_cuda=self.use_cuda)        

    def tpsPointTnf(self,theta,points):
        # points are expected in [B,3,N], where first row is X and second row is Y
        # reshape points for applying Tps transformation
        points=points.unsqueeze(3).unsqueeze(4).transpose(1,4)
        # apply transformation
        warped_points = self.tpsTnf.apply_transformation(theta,points)
        # undo reshaping
        #print("hahahah")
        warped_points=warped_points.transpose(4,1).contiguous().squeeze(4).squeeze(3)         
        return warped_points
    
    def affPointTnf(self,theta,points):
        theta_mat = theta.view(-1,2,3)
        warped_points = torch.bmm(theta_mat[:,:,:2],points)
        warped_points += theta_mat[:,:,2].unsqueeze(2).expand_as(warped_points)
        return warped_points 

def PointsToUnitCoords(P,im_size):
    h,w = im_size[:,0],im_size[:,1]
    NormAxis = lambda x,L: (x-1-(L-1)/2)*2/(L-1)
    P_norm = P.clone()
    # normalize Y
    P_norm[:,0,:] = NormAxis(P[:,0,:],w.unsqueeze(1).expand_as(P[:,0,:]))
    # normalize X
    P_norm[:,1,:] = NormAxis(P[:,1,:],h.unsqueeze(1).expand_as(P[:,1,:]))
    return P_norm

def PointsToPixelCoords(P,im_size):
    h,w = im_size[:,0],im_size[:,1]
    NormAxis = lambda x,L: x*(L-1)/2+1+(L-1)/2
    P_norm = P.clone()
    # normalize Y
    P_norm[:,0,:] = NormAxis(P[:,0,:],w.unsqueeze(1).expand_as(P[:,0,:]))
    # normalize X
    P_norm[:,1,:] = NormAxis(P[:,1,:],h.unsqueeze(1).expand_as(P[:,1,:]))
    return P_norm