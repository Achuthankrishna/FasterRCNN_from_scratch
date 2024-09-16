#This code will implement the backbone region proposal network for FastRCNN
import torch
import torch.nn as nn
import torch.signal
import torchvision
import numpy as np
import math
from utils import apply_reg_pred,clamp_box
#check device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RPN (nn.Module):
    """Class RPN : Region Proposal Network

    Args:
        nn (module): Takes input channels present in feature map
    """
    def __init__(self,in_ch=512):
        super(RPN,self).__init__()
        #define ratio for anchor box scales (H W C)
        self.scales=[128,256,512] #From paper
        self.asp_ratio=[0.5,1,2]
        self.num_anchors=len(self.scales)*len(self.asp_ratio)

        #model 
        
        """
        (3x3 conv -> 1x1 box clas layer -> K channels
                -> 1x1 box reg layer
        """
        
        self.conv=nn.Conv2d(in_ch,in_ch,kernel_size=3,stride=1,padding=1)

        self.box_cls=nn.Conv1d(in_ch,self.num_anchors,kernel_size=1,stride=1)
        self.box_reg=nn.Conv1d(in_ch,self.num_anchors*4,kernel_size=1,stride=1)



    def gen_anchors(self,image,feat):
        img_h,img_w=image.shape[-2:]
        fmap_h,fmap_w=feat.shape[-2:]

        #calculating stride in 2 dimensions 
        str1=torch.tensor(img_h//fmap_h,dtype=torch.int64)
        str2=torch.tensor(img_w//fmap_w,dtype=torch.int64)

        #Feature map putput will be VGG16 output of the image without lat layer
        scale=torch.tensor(self.scale,dtype=feat.dtype)
        asp_ratio=torch.tensor(self.asp_ratio,dype=feat.dtype)

        #creating anchor box of area 1 = h*w .. h/w=asp_ratio h=sqrt(asp_ratio) and w=1/h
        h_ratio=torch.sqrt(asp_ratio)
        w_ratio=1/h_ratio

        wid=(w_ratio[:,None]*scale[None,:]).view(-1)
        heigh=(h_ratio[:,None]*scale[None,:]).view(-1)

        #creating zero centered base anchor with unit area
        base_anch=torch.stack([-wid,-heigh,wid,heigh],dim=1)/2
        base_anch=base_anch.round() #Decimal

        #get grid of anchors by using shifts of strides on base anchor
        sh_x=torch.arange(0,fmap_w,dtype=torch.int32,device=feat.device) + str1 #shift x1 and x2
        sh_y=torch.arange(0,fmap_h,dtype=torch.int64,device=feat.device)+str2 #shift y1 and y2 

        sh_y,sh_x=torch.meshgrid(sh_y,sh_x,indexing='ij')

        sh_x=sh_x.reshape(-1)
        sh_y=sh_y.reshape(-1)
        #Box shifted
        shift= torch.stack((sh_x,sh_y,sh_x,sh_y),dim=1)
        #sh -> (H*W,4) base_anch=(num_anc_per_loc,4)
        anchor=(shift.view(-1,1,4)+base_anch(1,-1,4))
        anchors=anchor.reshape(-1,4) #(fmap_w*fmapw*h*9*4)
        return anchors


    def forward (self,image,fmap,target):
        """_summary_

        Args:
            image (1xHxWxC): input image of batch size 1
            fmap (_type_): Feature map with 512 channels
            target (dict): Target dictionary with bbox and labels (GT)
        """
        rpn_f=nn.ReLU()(self.conv(fmap))
        score=self.box_cls(rpn_f)
        bbox_reg=self.box_reg(rpn_f)

        #generating anchors
        anchor=self.gen_anchors(image,fmap)
        #redhspe scores as to the num of anchors we get
        num_anch=score.size(1)
        score=score.permute(0,2,3,1)
        score=score.reshape(-1,1) #score = (batch*fmap_h*num_anch,1)

        bbox_reg=bbox_reg.view(bbox_reg.size(0),num_anch,4,rpn_f.shape[-2],rpn_f.shape[-1])
        bbox_reg=bbox_reg.permute(0,3,4,1,2)
        bbox_reg=bbox_reg.reshape(-1,4) #score = (batch*fmap_h*num_anch,4)

        #transforming generated anchor box acc to bbox_transofrm_pred
        prop=apply_reg_pred(bbox_reg.detach().reshape(-1,1,4),anchor)
        prop=prop.reshape(prop.size(0),4)
        proposal,score=self.filter(prop,score.detach(),image.shape)

        rpn_out={"proposals":proposal,"scores":score}
        if target is None:
            return rpn_out
        else:
        #In training mode - assign GT and compute class and localization loss using box trans params and predicted scores
    def assign_targets(self,anchor,gt):
        # Get (GTBOX,num anchros)




        


    #Applying filter : Non max suppression as Fast RCNN
    def filter(self,proposal,class_score,img_shape):
        class_score=class_score.reshape(-1)
        class_score=torch.sigmoid(class_score)
        _,n_idx=class_score.topk(10000) #return top nk values
        class_score=class_score[n_idx]
        proposal=class_score[n_idx]
        proposal=clamp_box(proposal,img_shape)

        #NMS 
        mas=torch.zeros_like(class_score,dtype=torch.bool)
        keep_index=torchvision.ops.nms(proposal,class_score,0.7)
        nms_indices=keep_index[class_score[keep_index].sort(descending=True)[1]]
        proposal=proposal[nms_indices[:2000]] #training 2000 proposals as per paper
        class_score=class_score[nms_indices[:2000]] #training 2000 proposals as per paper
        return proposal,class_score



        


#We take the generated and apply predictions to all anchors which will tranform anchor boxes to proposal boxes using apply regression predictions


    







