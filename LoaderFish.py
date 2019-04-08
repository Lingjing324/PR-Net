import torch
import pickle
import os
import numpy as np
import matplotlib.image as img
import numpy as np
import cv2
import glob
import numpy as np
from operator import itemgetter 
from torch.utils.data import Dataset
import geotnf.transformation
import torch
import geotnf.point_tnf
import tqdm
from scipy.spatial.distance import squareform,pdist

def Tps_trans(Y,theta):
    YY=np.transpose(Y,[0,2,1])
    TT=torch.Tensor(theta.astype(np.float32))
    YY=torch.Tensor(YY.astype(np.float32))
    a=geotnf.point_tnf.PointTnf(use_cuda=True)
    aa=a.tpsPointTnf(TT,YY).cpu().numpy()
    return aa


class PointRegDataset(Dataset):
    def __init__(self, 
                 point_size=91,
                 total_data=100000, 
                 deform_level=0.4,
                 noise_ratio=0,
                 outlier_ratio=0, 
                 outlier_s=False,
                 outlier_t=False,
                 missing_points=0,   
                 geometric_model="tps",
                 miss_targ=False,
                 miss_source=False,
                noise_s=False,
                noise_t=False,
                clas=1):
        
        self.deform_level=deform_level
        self.noise_ratio=noise_ratio
        self.outlier_ratio=int(outlier_ratio*point_size)
        self.missing_points=missing_points
        self.geometric_model=geometric_model
        
        
        if clas==1:
            sc=[-0.9154191606171814, -0.16535078775508855, -0.890508968073163, -0.10212842773109136, -0.8572953780144712, -0.10212842773109136, -0.8240817879557792, -0.1495451977490876, -0.8739021730438176, -0.19696196776709046, -0.923722558131855, -0.16535078775508855, -1.0648803158812945, -0.2917955078030896, -1.02336332830793, -0.16535078775508855, -0.9818463407345652, -0.05471165771308847, -0.9569361481905461, 0.024316292316909686, -0.9154191606171814, 0.11914983235291547, -0.8822055705584902, 0.18237219237691266, -0.8074749929264337, 0.26140014240691745, -0.7576546078383953, 0.34042809243691563, -0.674620632691666, 0.4668728124849167, -0.5832832600302639, 0.5459007625149215, -0.5168560799128809, 0.6091231225389186, -0.4006085147074595, 0.7039566625749244, -0.3341813345900757, 0.8145957926169245, -0.26775415447269185, 1.1623187727489324, -0.201326974355308, 1.4626249828629307, -0.15150658926727137, 1.7629311929769422, -0.1432031917525986, 1.8893759130249435, -0.11829299920858027, 2.1106541731089434, -0.10168620417923308, 1.9684038630549416, -0.0933828066645603, 1.7471256029709414, -0.0933828066645603, 1.5416529328929356, -0.08507940914988753, 1.3045690828029344, -0.0933828066645603, 1.1149020027309295, -0.07677601163521475, 0.9094293326529237, -0.07677601163521475, 0.7197622525809254, -0.043562421576523666, 0.5617063525209225, -0.03525902406184923, 0.48267840249091765, 0.04777495108487849, 0.40365045246091946, 0.11420213120226233, 0.34042809243691563, 0.16402251629030062, 0.27720573241291846, 0.2221462988930117, 0.21398337238891457, 0.29687687652506667, 0.1033442423469145, 0.36330405664245047, -0.007294887695085575, 0.44633803178917986, -0.08632283772509039, 0.504461814391891, -0.05471165771308847, 0.5708889945092748, 0.04012188232291065, 0.6041025845679658, 0.18237219237691266, 0.6705297646853497, 0.3878448624549185, 0.7286535472880591, 0.6249287125449197, 0.8199909199494629, 0.6723454825629225, 0.9030248950961923, 0.7671790225989217, 0.8864181000668467, 0.6249287125449197, 0.8698113050375013, 0.4668728124849167, 0.8615079075228268, 0.26140014240691745, 0.8365977149788085, 0.1033442423469145, 0.8449011124934812, -0.1337396077430933, 0.8449011124934812, -0.30760109780909056, 0.8449011124934812, -0.5288793578930974, 0.8781147025521724, -0.7343520279710966, 0.9279350876402106, -0.8924079280310996, 0.9860588702429217, -1.0030470580731063, 1.03587925533096, -1.1294917781211073, 1.044182652845631, -1.2085197281511055, 0.9694520752135745, -1.1452973681271084, 0.8864181000668467, -1.0504638280911025, 0.7950807274054447, -0.9714358780611043, 0.6871365597146952, -0.9240191080431015, 0.5957991870532932, -0.8291855680071023, 0.537675404450582, -0.7501576179770976, 0.47124822433319985, -0.6395184879350975, 0.4131244417304888, -0.5762961279111003, 0.2885734790103939, -0.5288793578930974, 0.18893270883431895, -0.46565699786909354, 0.08929193865824402, -0.46565699786909354, -0.03525902406184923, -0.49726817788109545, -0.08507940914988753, -0.49726817788109545, -0.0020454340031581387, -0.5762961279111003, 0.0975953361729168, -0.6553240779410984, 0.1723259138049734, -0.6869352579531003, 0.26366328646637555, -0.7343520279710966, 0.21384290137833725, -0.7343520279710966, 0.03947155357020572, -0.7343520279710966, -0.15150658926727137, -0.7343520279710966, -0.26775415447269185, -0.7185464379651023, -0.4006085147074595, -0.6711296679470994, -0.46703569482484336, -0.6079073079230956, -0.5583730674862455, -0.5288793578930974, -0.5998900550596102, -0.5130737678870964, -0.7742614028677418, -0.46565699786909354, -0.890508968073163, -0.4814625878750945, -0.9818463407345652, -0.43404581785709156, -1.0399701233372756, -0.3708234578330944, -1.0565769183666218, -0.2917955078030896, -0.05048191950541625, -0.7580604129800946, 0.09967118555158623, -0.7580604129800946]
            source=np.asarray(sc).reshape(-1,2)

        if clas==2:
            source=(np.loadtxt('./data/hand2.txt')
).reshape(-1,2)
            source=source[:256,:]      
        if clas==3:
            source=(np.loadtxt('./data/human_skeleton.txt')
).reshape(-1,2)
            source=source[:256,:]

        if clas==4:
            source=(np.loadtxt('./data/skull_try_1.txt')
).reshape(-1,2)
            source=source[:256,:]
            
        ################################
        SC=[]
        Targ=[]
        Theta=[]
        Targ_clean=[]        
        source1=source        
        
        for i in tqdm.tqdm(range(total_data)): 

            theta = np.array([-1 , -1 , -1 , 0 , 0 , 0 , 1 , 1 , 1 , -1 , 0 , 1 , -1 , 0 , 1 , -1 , 0 , 1])
            theta = theta+(np.random.rand(18)-0.5)*2*deform_level   
            targ=Tps_trans(np.tile(np.expand_dims(source,0),[2,1,1]),
             np.tile(np.expand_dims(theta,0),[2,1]))
            
            targ=targ[0]
            targc=targ
            
            if self.noise_ratio!=0:
                if noise_s:
                    source=source+np.random.normal(0,self.noise_ratio,source.shape)
                if noise_t:
                    targ=targ+np.random.normal(0,self.noise_ratio,targ.shape)
            
            if self.outlier_ratio!=0:
                
                if outlier_s:
                    addi=np.asarray([np.random.uniform(-2,2,self.outlier_ratio),
                         np.random.uniform(-2,2,self.outlier_ratio)]).T

                    source=np.concatenate([source,addi], axis=0)
            
                if outlier_t:
                    addi=np.asarray([np.random.uniform(-2,2,self.outlier_ratio),
                                     np.random.uniform(-2,2,self.outlier_ratio)]).T
                    targ=np.concatenate([targ.T, addi], axis=0).T
                    if i==0:
                        ind=np.random.choice(range(len(source)),self.outlier_ratio)
                        source1=np.concatenate([source,source[ind]], axis=0)                    
                    
            if self.missing_points!=0:
                if miss_source:
                    YY=source
                    Pdist=squareform(pdist(YY))
                    selectingPoints=np.argsort(Pdist)[np.random.choice(
                        range(point_size))][self.missing_points:]
                    source=YY[selectingPoints]
                    
                if miss_targ:
                    YY=targ.T
                    Pdist=squareform(pdist(YY))
                    selectingPoints=np.argsort(Pdist)[np.random.choice(
                        range(point_size))][self.missing_points:]
                    targ=YY[selectingPoints]
                    ind=np.random.choice(range(len(targ)),self.missing_points)
                    targ=np.concatenate([targ,targ[ind]], axis=0)

            Targ_clean.append(targc)
            SC.append(source1.T)            
            Theta.append(theta)            
            Targ.append(targ)

        self.source_list=SC
        self.theta_list=Theta
        self.target_list=Targ
        self.target_clean_list=Targ_clean
        
        
    def __getitem__(self, index):
        target=self.target_list[index]
        source=self.source_list[index]
        theta=self.theta_list[index]    
        tc=self.target_clean_list[index]
        return target,source,theta, tc
    
    
    def __len__(self):
        return len(self.theta_list)
