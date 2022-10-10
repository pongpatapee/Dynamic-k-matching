import tensorflow as tf
import numpy as np

from dynamic_k_matching_tf import dynamic_k_matching

"""Testing YOLOX Dynamic_k Matching output with outputs from Pytorch"""
class DynamicKMatchingTest(tf.test.TestCase):
    # size of gt_class should be <= num cols
    # size of fg_mask has to match num cols

    def test_tf_torch_diff_dynamic_k_square_mat(self):
        np.random.seed(0)
        cost = np.random.uniform(0.0, 1.0, (8, 8))
        pair_wise_ious = np.random.uniform(0.0, 1.0, (8, 8))
        gt_classes = np.random.uniform(0.0, 100.0, 8)
        num_gt = 8
        fg_mask = np.full(8, True)

        cost = tf.cast(tf.constant(cost), tf.dtypes.float32)
        pair_wise_ious = tf.cast(tf.constant(
            pair_wise_ious), tf.dtypes.float32)
        gt_classes = tf.cast(tf.constant(gt_classes), tf.dtypes.float32)
        fg_mask = tf.cast(tf.constant(fg_mask), tf.dtypes.float32)
        res_tf = dynamic_k_matching(
            cost, pair_wise_ious, gt_classes, num_gt, fg_mask)

        num_fg = int(res_tf[0].numpy())
        gt_matched_classes = res_tf[1].numpy()
        pred_ious_this_matching = res_tf[2].numpy()
        matched_gt_inds = res_tf[3].numpy()

        expected_num_fg = 8
        expected_gt_matched_classes = np.array(
            [61.801544, 59.087276, 13.547406, 29.828232, 59.087276, 56.99649,  29.007761, 29.007761], dtype=np.float32)
        expected_pred_ious_this_matching = np.array(
            [0.31798318, 0.50132436, 0.82894003, 0.5722519, 0.42385504, 0.5812729, 0.2961402, 0.11872772, ], dtype=np.float32)
        expected_matched_gt_inds = np.array([2, 7, 4, 5, 7, 6, 1, 1])

        self.assertEqual(num_fg, expected_num_fg)
        self.assertAllEqual(gt_matched_classes, expected_gt_matched_classes)
        self.assertAllEqual(pred_ious_this_matching,
                            expected_pred_ious_this_matching)
        self.assertAllEqual(matched_gt_inds, expected_matched_gt_inds)

    def test_tf_torch_diff_dynamic_k_nonsquare_mat(self):
        np.random.seed(0)
        cost = np.random.uniform(0.0, 1.0, (8, 10))
        pair_wise_ious = np.random.uniform(0.0, 1.0, (8, 10))
        gt_classes = np.random.uniform(0.0, 100.0, 10)
        num_gt = 8
        fg_mask = np.full(10, True)

        cost = tf.cast(tf.constant(cost), tf.dtypes.float32)
        pair_wise_ious = tf.cast(tf.constant(
            pair_wise_ious), tf.dtypes.float32)
        gt_classes = tf.cast(tf.constant(gt_classes), tf.dtypes.float32)
        fg_mask = tf.cast(tf.constant(fg_mask), tf.dtypes.float32)
        res_tf = dynamic_k_matching(
            cost, pair_wise_ious, gt_classes, num_gt, fg_mask)

        num_fg = int(res_tf[0].numpy())
        gt_matched_classes = res_tf[1].numpy()
        pred_ious_this_matching = res_tf[2].numpy()
        matched_gt_inds = res_tf[3].numpy()

        expected_num_fg = 10
        expected_gt_matched_classes = np.array(
            [1.1714084,  1.1714084, 86.63823,   97.55215,   86.63823,   35.997807, 45.354267,   1.1714084, 35.997807,   1.1714084], dtype=np.float32)
        expected_pred_ious_this_matching = np.array(
            [0.8965466,  0.36756188, 0.6994793,  0.6439902,  0.81379783, 0.8480082, 0.5865129,  0.9194826,  0.4071833,  0.998847], dtype=np.float32)
        expected_matched_gt_inds = np.array([6, 6, 3, 4, 3, 7, 1, 6, 7, 6])

        self.assertEqual(num_fg, expected_num_fg)
        self.assertAllEqual(gt_matched_classes, expected_gt_matched_classes)
        self.assertAllEqual(pred_ious_this_matching,
                            expected_pred_ious_this_matching)
        self.assertAllEqual(matched_gt_inds, expected_matched_gt_inds)


if __name__ == "__main__":
    tf.test.main()
