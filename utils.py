import numpy as np
import torch


def apply_reg_pred(bbox_trans_pred,anchors):
    """_summary_

    Args:
        bbox_trans_pred (_type_): (num_anchors,num_class,4)
        anchors (_type_): (num_anchors,4)

    return:
        prediction_box :(num_anchors,num_class,4)
    """
    bbox_trans_pred=bbox_trans_pred.reshape(bbox_trans_pred.size(0),-1,4)
    #Get H and W from x1 x2 y1 y2
    w=anchors[:,2]-anchors[:,0]
    h=anchors[:,3]-anchors[:,1]
    cx=anchors[:,0] + 0.5 *w #as per paper
    cy=anchors[:,1] + 0.5 *h #as per paper

    dx = bbox_trans_pred[:, :, 0]
    dy = bbox_trans_pred[:,:,1]
    dw = bbox_trans_pred[:,:,2]
    dh = bbox_trans_pred[:,:,3]

    #dh=(num_anchors,numclasses)
    pred_center_x = dx + w[:,None] + cx[:,None]
    pred_center_y = dy + h[:,None] + cy[:,None]
    pred_h= torch.exp(dh) *h[:,None]
    pred_w= torch.exp(dw) *w[:,None]

    pred_box_x1=pred_center_x - 0.5 * pred_w
    pred_box_y1=pred_center_y - 0.5 * pred_h
    pred_box_x2=pred_center_x + 0.5 * pred_w
    pred_box_y2=pred_center_y + 0.5 * pred_h

    pred_box=torch.stack((pred_box_x1,pred_box_y1,pred_box_x2,pred_box_y2),dim=2)
    return pred_box

def clamp_box(box,img_shape):
    box_x1=box[:,:,0]
    box_y1=box[:,:,1]
    box_x2=box[:,:,2]
    box_y2=box[:,:,3]
    h,w=img_shape[-2,:]
    box_x1=box_x1.clamp(min=0,max=w)
    box_x2=box_x2.clamp(min=0,max=w)
    box_y1=box_y1.clamp(min=0,max=h)
    box_y2=box_y2.clamp(min=0,max=h)
    box=torch.cat((box_x1[:,:,None],box_y1[:,:,None],box_x2[:,:,None],box_y2[:,:,None]),dim=-1)
    return box

def get_iou(b1,b2):
    """_summary_

    Args:
        b1 (list): box coords of shape N x 4
        b2 (list): pred box coords M x 4
    Return:
        IOU Matrix of shape (N x M)
    """
    # Area = x2-x1 * y2-y1
    area1=(b1[:,2]-b1[:,0])*(b1[:,3]-b1[:,1])
    area2=(b2[:,2]-b2[:,0])*(b2[:,3]-b2[:,1])
    #get top x1,y1
    x_left = torch.max(b1[:,None,0],b2[:,0])
    y_top = torch.max(b1[:,None,1],b2[:,1])
    #get right and bottom x2 y2
    x_right=torch.min(b1[:,None,2],b2[:,2])
    y_bott=torch.min(b1[:,None,3],b2[:,3])

    iou_area=(x_right-x_left).clamp(min=0) * (y_bott-y_top).clamp(min=0)
    union_area=area1[:,None]+area2 -iou_area
    return iou_area/union_area




    
