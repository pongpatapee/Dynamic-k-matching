import torch
#Pytorch's version
def dynamic_k_matching_pytorch(cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
      # Dynamic K
      # ---------------------------------------------------------------
      matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
      ious_in_boxes_matrix = pair_wise_ious
      n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
      topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
      dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
      dynamic_ks = dynamic_ks.tolist()
      for gt_idx in range(num_gt):
          _, pos_idx = torch.topk(
              cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
          )
          matching_matrix[gt_idx][pos_idx] = 1

      del topk_ious, dynamic_ks, pos_idx

      anchor_matching_gt = matching_matrix.sum(0)
      if (anchor_matching_gt > 1).sum() > 0:
          _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
          matching_matrix[:, anchor_matching_gt > 1] *= 0
          matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1

      fg_mask_inboxes = matching_matrix.sum(0) > 0
      num_fg = fg_mask_inboxes.sum().item()
      fg_mask[fg_mask.clone()] = fg_mask_inboxes
      matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
      gt_matched_classes = gt_classes[matched_gt_inds]
      pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
          fg_mask_inboxes
      ]
      return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds