import numpy as np
import os
import cv2
from random import sample



def vis_symmetry(sym_cor_pred, mask_pred, sym_cor, mask, image, i_batch):
    """
    visualize symmetry
    :param sym_cor_pred: 2 * 480 * 640
    :param mask_pred: 1 * 480 * 640
    :param sym_cor: 2 * 480 * 640 (symmetric GT)
    :param mask: 1 * 480 * 640 (maks GT)
    :param image: 480 * 640 * 3
    :param epoch: epoch (500)
    :param i_batch: batch
    :return:
    """
    img_dir = os.path.join("/home/hwanglab/HybridPose/test_output", 'img_symmetry')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    # visualize prediction
    image_pred = image.copy()

    sym_cor_pred = sym_cor_pred.detach().cpu().numpy()
    ys, xs = np.nonzero(mask_pred[0])
    for i_pt in sample([i for i in range(len(ys))], min(100, len(ys))):
        y = int(round(ys[i_pt]))
        x = int(round(xs[i_pt]))
        x_cor, y_cor = sym_cor_pred[:, y, x]
        x_cor = int(round(x + x_cor))
        y_cor = int(round(y + y_cor))
        image_pred = cv2.line(image_pred, (x, y), (x_cor, y_cor), (0, 0, 255), 1)
    img_pred_name = os.path.join(img_dir, '{}_test_sym.jpg'.format(str(i_batch[0])))
    cv2.imwrite(os.path.join(img_dir, "{}_test_original_img.jpg").format(str(i_batch[0])),image)
    cv2.imwrite(img_pred_name, image_pred)
    # visualize ground truth
    image_gt = image.copy()
    mask = mask.detach().cpu().numpy()[0]
    sym_cor = sym_cor.detach().cpu().numpy()
    ys, xs = np.nonzero(mask)  # mask 480 * 640
    for i_pt in sample([i for i in range(len(ys))], min(100, len(ys))):
        y = int(round(ys[i_pt]))
        x = int(round(xs[i_pt]))
        x_cor, y_cor = sym_cor[:, y, x]
        x_cor = int(round(x + x_cor))
        y_cor = int(round(y + y_cor))
        image_gt = cv2.line(image_gt, (x, y), (x_cor, y_cor), (0, 255, 0), 1)
    img_gt_name = os.path.join(img_dir, '{}_test_sym_gt.jpg'.format(str(i_batch[0])))
    cv2.imwrite(img_gt_name, image_gt)


def vis_mask(mask_pred, mask, i_batch):
    mask_pred = mask_pred[0]
    mask = np.uint8(mask.detach().cpu().numpy()[0])
    image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    # red: prediction
    image[mask_pred == 1.] += np.array([0, 0, 128], dtype=np.uint8)
    # blue: gt
    image[mask != 0] += np.array([128, 0, 0], dtype=np.uint8)
    img_dir = os.path.join("/home/hwanglab/HybridPose/test_output", 'img_mask')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img_name = os.path.join(img_dir, '{}_mask.jpg'.format(str(i_batch[0])))
    cv2.imwrite(img_name, image)


def vis_keypoints(pts2d_map_pred, pts2d, image, i_batch):
    img_dir = os.path.join("/home/hwanglab/HybridPose/test_output", 'img_keypoints')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    # vote keypoints
    pts2d_pred = pts2d_map_pred.detach().cpu().numpy()
    # draw predication
    image_pred = image.copy()
    for i in range(pts2d_pred.shape[0]):
        x, y = pts2d_pred[i]
        x = int(round(x))
        y = int(round(y))
        # radius=2, color=red, thickness=filled
        image_pred = cv2.circle(image_pred, (x, y), 4, (0, 0, 255), thickness=-1)
    # img_pred_name = os.path.join(img_dir, '{}_2D_keypoints.jpg'.format(str(i_batch[0])))
    # cv2.imwrite(img_pred_name, image_pred)
    # draw ground truth
    pts2d = pts2d.detach().cpu().numpy()
    image_gt = image.copy()
    for i in range(pts2d.shape[0]):
        x, y = pts2d[i]
        x = int(round(x))
        y = int(round(y))
        # radius=2, color=white, thickness=filled
        image_gt = cv2.circle(image_pred, (x, y), 4, (0, 255, 0), thickness=-1)
    img_gt_name = os.path.join(img_dir, '{}_2D_keypoints.jpg'.format(str(i_batch[0])))
    cv2.imwrite(img_gt_name, image_gt)


def vis_graph(graph_pred, graph_gt, pts2d_gt, mask_pred, mask_gt, image, i_batch):
    img_dir = os.path.join("/home/hwanglab/HybridPose/test_output", 'img_graph')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    image_gt = image.copy()
    image_pred = image.copy()
    graph_pred = graph_pred.detach().cpu().numpy()
    graph_pred = graph_pred.reshape((-1, 2, image.shape[0], image.shape[1]))
    graph_gt = graph_gt.detach().cpu().numpy()
    graph_gt = graph_gt.reshape((-1, 2, image.shape[0], image.shape[1]))
    pts2d_gt = pts2d_gt.numpy()
    mask_pred = mask_pred[0]
    mask_gt = mask_gt.detach().cpu().numpy()[0]
    num_pts = pts2d_gt.shape[0]
    i_edge = 0
    for start_idx in range(0, num_pts - 1):
        for end_idx in range(start_idx + 1, num_pts):
            # pred, red
            start = np.int16(np.round(pts2d_gt[start_idx]))
            edge_x = graph_pred[i_edge, 0][mask_pred == 1.].mean()
            edge_y = graph_pred[i_edge, 1][mask_pred == 1.].mean()
            edge = np.array([edge_x, edge_y])
            end = np.int16(np.round(pts2d_gt[start_idx] + edge))
            image_pred = cv2.line(image_pred, tuple(start), tuple(end), (0, 0, 255), 1)
            # gt, green
            start = np.int16(np.round(pts2d_gt[start_idx]))
            edge_x = graph_gt[i_edge, 0][mask_gt == 1.].mean()
            edge_y = graph_gt[i_edge, 1][mask_gt == 1.].mean()
            edge = np.array([edge_x, edge_y])
            end = np.int16(np.round(pts2d_gt[start_idx] + edge))
            image_gt = cv2.line(image_gt, tuple(start), tuple(end), (0, 255, 0), 1)
            i_edge += 1
    img_gt_name = os.path.join(img_dir, '{}_graph_gt.jpg'.format(str(i_batch[0])))
    cv2.imwrite(img_gt_name, image_gt)
    img_pred_name = os.path.join(img_dir, '{}_graph_pred.jpg'.format(str(i_batch[0])))
    cv2.imwrite(img_pred_name, image_pred)


