#### this script is just a sandbox to test random stuff
from more_itertools import pairwise
import numpy as np
import tensorflow as tf
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

from dynamic_k_tf import dynamic_k_matching
from dynamic_k_torch import dynamic_k_matching_pytorch

def test_tf_torch_diff_dynamic_k():
    cost = np.random.uniform(0.0, 1.0, (8, 10))
    pair_wise_ious = np.random.uniform(0.0, 1.0, (8, 10))
    gt_classes = np.random.uniform(0.0, 50.0, 10)
    num_gt = 8
    # fg_mask = np.random.choice([True, False], size=(8))
    fg_mask = np.full(8, True)

    # cost = np.ones((8, 8))
    # pair_wise_ious = np.ones((8, 8))
    # gt_classes = np.random.uniform(0, 50, 8)
    # num_gt = 8
    # fg_mask = np.full(8, True)

    # cost = np.random.uniform(0.0, 1.0, (8, 8))
    # pair_wise_ious = np.random.uniform(0.0, 1.0, (8, 8))
    # gt_classes = np.random.uniform(0.0, 100.0, 8)
    # num_gt = 8
    # fg_mask = np.full(8, True)

    res_torch = dynamic_k_matching_pytorch(torch.tensor(cost), torch.tensor(pair_wise_ious), torch.tensor(gt_classes), num_gt, torch.tensor(fg_mask))

    cost = tf.cast(tf.constant(cost), tf.dtypes.float32)
    pair_wise_ious = tf.cast(tf.constant(pair_wise_ious), tf.dtypes.float32)
    gt_classes = tf.cast(tf.constant(gt_classes), tf.dtypes.float32)
    fg_mask = tf.cast(tf.constant(fg_mask), tf.dtypes.float32)
    res_tf = dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
    
    num_fg_torch = res_torch[0]
    gt_matched_classes_torch = np.float32(res_torch[1].numpy())
    pred_ious_this_matching_torch = np.float32(res_torch[2].numpy()) #np.array(res_torch[2])
    matched_gt_inds_torch  = np.float32(res_torch[3].numpy()) #np.array(res_torch[3])

    num_fg_tf = int(res_tf[0].numpy())
    gt_matched_classes_tf = res_tf[1].numpy()
    pred_ious_this_matching_tf = res_tf[2].numpy()
    matched_gt_inds_tf = res_tf[3].numpy()

    print("Pytorch output converted to numpy: ")
    print(num_fg_torch, gt_matched_classes_torch, pred_ious_this_matching_torch, matched_gt_inds_torch)
    print("TensorFlow output converted to numpy: ")
    print(num_fg_tf, gt_matched_classes_tf, pred_ious_this_matching_tf, matched_gt_inds_tf)
    # print(type(gt_matched_classes_torch[0]), type(gt_matched_classes_tf[0]))
    # print(res_tf)

# print(dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask))
test_tf_torch_diff_dynamic_k()