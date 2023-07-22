import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

def visualize_local_tracklets(poses,joints_num=17,scale_factor=100):

    poses = poses.reshape((-1,int(joints_num),2))
    tracklet_img = visualize_tracklet(poses,scale_factor,joints_num)

    return tracklet_img

def visualize_tracklet(tracklet,scale_factor,num_joints):

    imgs = []
    for pose in tracklet:
        img = visulize_single_pose(pose,scale_factor,num_joints)
        imgs.append(img)
    imgs = np.hstack(imgs)

    return imgs

def visulize_single_pose(kpts,scale_factor,num_joints):

    if num_joints == 17:
        links = [(0, 1), (0, 2), (1, 3), (2, 4),
                (5, 7), (7, 9), (6, 8), (8, 10),
                (11, 13), (13, 15), (12, 14), (14, 16),
                (3, 5), (4, 6), (5, 6), (5, 11), (6, 12), (11, 12)]

    if num_joints == 25:
        links = [(17,15),(15,0),(0,16),(16,18),(0,1),(1,8),(1,2),(2,3),(3,4),(1,5),(5,6),
            (6,7),(8,9),(9,10),(10,11),(11,22),(22,23),(11,24),(8,12),(12,13),(13,14),(14,21),(14,19),(19,20)]
    
    kpts = np.array(kpts)

    x = kpts[:,0]
    y = kpts[:,1]

    img = np.zeros((100,100,3),np.uint8)
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(links) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    for i in range(len(links)):

        order1, order2 = links[i][0], links[i][1]
        x1 =int(((np.float32(x[order1])))) + int(scale_factor/2)
        y1 =int(((np.float32(y[order1])))) + int(scale_factor/2)
        x2 =int(((np.float32(x[order2])))) + int(scale_factor/2)
        y2 =int(((np.float32(y[order2])))) + int(scale_factor/2)
        cv2.line(img,(x1,y1),(x2,y2),thickness=1,color=colors[i])

    return img