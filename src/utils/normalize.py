import numpy as np

def normalize_anomaly_score(scores):
    max_score = max(scores)
    min_score = min(scores)
    length = max_score - 0
    if length==0:
        length=1
    scores = np.array(scores)

    return scores/length

def normalize_score(score):

    return score/np.sum(score)

def normalize_pose(pose_x, pose_y):

    x_max, y_max = np.max(pose_x,axis=1), np.max(pose_y,axis=1)
    x_min, y_min = np.min(pose_x,axis=1), np.min(pose_y,axis=1)
    x_c, y_c = (x_max+x_min)/2, (y_max+y_min)/2
    w, h = x_max-x_min, y_max - y_min

    x, y = [], []

    for i in range(len(w)):
        nor_x = ((pose_x[i] - x_c[i]) / w[i]).tolist()
        nor_y = ((pose_y[i] - y_c[i]) / h[i]).tolist()
        x.append(nor_x)
        y.append(nor_y)

    return x, y

def center_pose(pose_x, pose_y):

    x_max, y_max = np.max(pose_x,axis=1), np.max(pose_y,axis=1)
    x_min, y_min = np.min(pose_x,axis=1), np.min(pose_y,axis=1)
    x_c, y_c = (x_max+x_min)/2, (y_max+y_min)/2
    w, h = x_max-x_min, y_max - y_min

    w[w<1e-5] = 1
    h[h<1e-5] = 1

    x, y = [], []

    for i in range(len(w)):
        nor_x = ((pose_x[i] - x_c[i])).tolist()
        nor_y = ((pose_y[i] - y_c[i])).tolist()
        x.append(nor_x)
        y.append(nor_y)

    return x, y

def keypoints17_to_coco18(kps):

    kp_np = np.array(kps)#.reshape(-1,17,3)
    neck_kp_vec = 0.5 * (kp_np[..., 5, :] + kp_np[..., 6, :])
    kp_np = np.concatenate([kp_np, neck_kp_vec[..., None, :]], axis=-2)
    opp_order = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    opp_order = np.array(opp_order, dtype=np.int)
    kp_coco18 = kp_np[..., opp_order, :]

    return kp_coco18