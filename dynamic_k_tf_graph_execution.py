import tensorflow as tf

#TensorFlow Version
@tf.function()
def dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask):

  matching_matrix = tf.zeros_like(cost, dtype=tf.uint8)
  flattened_idx = []
  ious_in_boxes_matrix = pair_wise_ious
  n_candidate_k = min(10, ious_in_boxes_matrix.shape[1])
  topk_ious, _ = tf.math.top_k(ious_in_boxes_matrix, k=n_candidate_k)
  dynamic_ks = tf.math.maximum(tf.cast(tf.reduce_sum(topk_ious, axis=1), tf.int32), [1])
  
  for gt_idx in range(num_gt):
      num_top = dynamic_ks[gt_idx]
      _, pos_idx = tf.math.top_k(
          -cost[gt_idx], k=num_top, sorted=False)
      

      indices = tf.TensorArray(tf.int32, size = 0, dynamic_size = True, clear_after_read = False)
      updates = tf.TensorArray(tf.int32, size = 0, dynamic_size = True, clear_after_read = False)
      for id in range(len(pos_idx)):
        indices = indices.write(id, [gt_idx, pos_idx[id]])
        updates = updates.write(id, 1)

      matching_matrix = tf.tensor_scatter_nd_update(matching_matrix, indices.stack(), tf.cast( updates.stack(), dtype = tf.uint8 ))

      indices.close()
      updates.close()

  del topk_ious, dynamic_ks, pos_idx

  anchor_matching_gt = tf.math.reduce_sum(matching_matrix, 0)
  

  if tf.math.reduce_sum(tf.cast((anchor_matching_gt > 1), tf.int32)) > 0:
    mask = anchor_matching_gt > 1
    cost_after_mask = tf.boolean_mask(cost, mask, axis=1)
    cost_argmin = tf.math.argmin(cost_after_mask)
    mask = tf.reshape(mask, [1, len(mask)])
    num_repeat = tf.constant([matching_matrix.shape[0], 1], tf.int32)
    tiled_mask = tf.tile(mask, num_repeat)
    indices = tf.where(tiled_mask)

    updates = tf.TensorArray(tf.int64, size = 0, dynamic_size = True, clear_after_read = False)
    for id in range(len(indices)):
      updates.stack()
      updates = updates.write(id, 0)
      
    matching_matrix = tf.tensor_scatter_nd_update(matching_matrix, indices, tf.cast(updates.stack(), dtype = tf.uint8))

    indices = tf.TensorArray(tf.int32, size = 0, dynamic_size= True, clear_after_read = False)
    updates = tf.TensorArray(tf.int32, size = 0, dynamic_size= True, clear_after_read = False)
    costargmin_ind = 0
    for id in range(len(mask[0])):
      if mask[0][id]:
        cost_ind = cost_argmin[costargmin_ind]
        index = [cost_ind, id]
        indices = indices.write(id, index)
        updates = updates.write(id, 1)
        costargmin_ind += 1
    matching_matrix = tf.tensor_scatter_nd_update(matching_matrix, indices.stack(), tf.cast( updates.stack(), dtype = tf.uint8 ))

  fg_mask_inboxes = tf.math.reduce_sum(matching_matrix, axis=0) > 0
  num_fg = tf.math.reduce_sum( tf.cast(fg_mask_inboxes, tf.float32) )
  fg_mask = tf.boolean_mask(fg_mask_inboxes, fg_mask)
  matched_gt_inds = tf.math.argmax(tf.boolean_mask(matching_matrix, fg_mask_inboxes, axis=1))
  gt_matched_classes = tf.gather(gt_classes, matched_gt_inds)
  pred_ious_this_matching_temp1 = tf.math.multiply(tf.cast(matching_matrix, tf.float32), pair_wise_ious)
  pred_ious_this_matching_temp2 = tf.math.reduce_sum(pred_ious_this_matching_temp1, axis=0) 
  pred_ious_this_matching = tf.boolean_mask(pred_ious_this_matching_temp2, fg_mask_inboxes)

  return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
