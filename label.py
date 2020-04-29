import argparse
import os
import cv2
import numpy as np
import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fx', type=float, default=572.41140)
    parser.add_argument('--fy', type=float, default=573.57043)
    parser.add_argument('--px', type=float, default=325.26110)
    parser.add_argument('--py', type=float, default=242.04899)
    parser.add_argument('--img_h', type=int, default=480)
    parser.add_argument('--img_w', type=int, default=640)
    parser.add_argument('--object_name', type=str, default='can')
    args = parser.parse_args()
    return args

def get_num_examples(object_name):
    return len(list(filter(lambda x: x.endswith('txt'), os.listdir(os.path.join('valid_poses', object_name)))))

def read_pose_and_img_id(object_name, example_id):
    filename = os.path.join('valid_poses', object_name, '{}.txt'.format(example_id))
    read_rotation = False
    read_translation = False
    R = []
    T = []
    with open(filename) as f:
        for line in f:
            if read_rotation:
                R.append(line.split())
                if len(R) == 3:
                    read_rotation = False
            elif read_translation:
                T = line.split()
                read_translation = False
            if line.startswith('rotation'):
                read_rotation = True
            elif line.startswith('center'):
                read_translation = True
    R = np.array(R, dtype=np.float32) # 3*3
    T = np.array(T, dtype=np.float32).reshape((3, 1)) # 3*1
    img_id = int(line) # in the last line
    return R, T, img_id

def read_3d_points(filename):
    with open(filename) as f:
        vertices = []
        for line in f:
            vertex = line.split()[:3]
            vertex = np.array([[float(vertex[0])],
                               [float(vertex[1])],
                               [float(vertex[2])]], dtype=np.float32)
            vertices.append(vertex)
    vertices = np.array(vertices, dtype=np.float32)
    return vertices

def parse_symmetry(filename):
    with open(filename) as f:
        lines = f.readlines()
        point = lines[1].split()
        point = (float(point[0]), float(point[1]), float(point[2]))
        normal = lines[3].split()
        normal = (float(normal[0]), float(normal[1]), float(normal[2]))
    return point, normal

def get_camera_intrinsic_matrix(args):
    return np.matrix([[args.fx, 0, args.px],
                      [0, args.fy, args.py],
                      [0, 0, 1]], dtype=np.float32)

def get_alignment_flipping():
    return np.matrix([[1., 0., 0.],
                      [0., -1., 0.],
                      [0., 0., -1.]])

def nearest_nonzero_idx_v2(a, x, y):
    # https://stackoverflow.com/questions/43306291/find-the-nearest-nonzero-element-and-corresponding-index-in-a-2d-array
    # x: (N,)
    # y: (N,)
    tmp = a[x, y]
    a[x, y] = 0
    r, c = np.nonzero(a)
    r = r.reshape((1, -1)).repeat(x.shape[0], axis=0)
    c = c.reshape((1, -1)).repeat(x.shape[0], axis=0)
    a[x, y] = tmp
    min_idx = ((r - x.reshape((-1, 1))) ** 2 + (c - y.reshape((-1, 1))) ** 2).argmin(axis=1)
    return np.array([r[0, min_idx], c[0, min_idx]]).transpose()

def fill(im_in):
    # based on https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
    # Threshold.
    # Set values equal to or above 220 to 0.
    # Set values below 220 to 255.
    th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY)
    # Copy the thresholded image.
    im_floodfill = im_th.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    return im_out

def main():
    args = parse_args()
    K = get_camera_intrinsic_matrix(args)
    F_flip = get_alignment_flipping()
    P_list = read_3d_points(os.path.join('models', args.object_name, '004.xyz'))
    O, n = parse_symmetry(os.path.join('my_labels', args.object_name, 'symmetries.txt'))
    keypts_3d = np.load(os.path.join('my_labels', args.object_name, 'keypoints.npy'))
    keypts_2d = np.zeros((get_num_examples(args.object_name), keypts_3d.shape[0], 2), dtype=np.float32)

    # for each 3D point P, find its correspondence P'
    P_prime_list = []
    for P_idx, P in enumerate(P_list):
        PO = (O[0] - P[0], O[1] - P[1], O[2] - P[2])
        dot_product = PO[0] * n[0] + PO[1] * n[1] + PO[2] * n[2]
        P_prime = (P[0] + 2 * dot_product * n[0], P[1] + 2 * dot_product * n[1], P[2] + 2 * dot_product * n[2])
        P_prime_list.append(P_prime)
    P_prime_list = np.array(P_prime_list)
 
    def project(P, R, T):
        P_RT = R * P + T
        p = K * P_RT
        x = int(round(p[0, 0] / p[2, 0]))
        y = int(round(p[1, 0] / p[2, 0]))
        return (x, y, P_RT[2, 0])

    for example_id in range(get_num_examples(args.object_name)):
        R, T, img_id = read_pose_and_img_id(args.object_name, example_id)
        R = F_flip * R
        T = F_flip * T
        # project 3D correspondeces to 2D
        correspondences = np.zeros((args.img_h, args.img_w, 2), dtype=np.int16)
        z_buffer = np.zeros((args.img_h, args.img_w), dtype=np.float32)
        is_filled = np.zeros((args.img_h, args.img_w), dtype=np.uint8)
        sample = example_id == 0
        if sample:
            img = cv2.imread('RGB-D/rgb_noseg/color_{:05d}.png'.format(img_id))
        mask = cv2.imread(os.path.join('masks', args.object_name,'{:d}.png'.format(img_id)))
        for P_idx, P in enumerate(P_list):
            P_prime = P_prime_list[P_idx]
            (x, y, z) = project(P, R, T)
            (x_prime, y_prime, _) = project(P_prime, R, T)
            if y >= 0 and y < args.img_h and x >= 0 and x < args.img_w:
                if is_filled[y, x] == 0 or z_buffer[y, x] > z:
                    # I did a simple experiment: a smaller z is closer to the camera than a bigger z
                    is_filled[y, x] = 1
                    z_buffer[y, x] = z
                    delta_x = x_prime - x
                    delta_y = y_prime - y
                    correspondences[y, x, 0] = delta_x
                    correspondences[y, x, 1] = delta_y
                    if sample and P_idx % 50 == 0:
                        # 1 sample every 50 points
                        # color: red, thickness: 1
                        img = cv2.line(img, (x, y), (x_prime, y_prime), (0, 0, 255), 1)

        mask = fill(mask)
        yx = np.argwhere((mask != 0) & (is_filled == 0.))
        yx_ = nearest_nonzero_idx_v2(is_filled, yx[:, 0], yx[:, 1])
        for i in range(yx.shape[0]):
            y, x = yx[i]
            y_, x_ = yx_[i]
            correspondences[y, x] = correspondences[y_, x_]
        os.makedirs(os.path.join('my_labels', args.object_name, 'cor'), exist_ok=True)
        # np.save(os.path.join('my_labels', args.object_name, 'cor', str(example_id) + '.npy'), correspondences)
        if sample:
            cv2.imwrite(os.path.join('my_labels', args.object_name, 'sample_correspondence.jpg'), img)
        # project 3D keypoints to 2D
        for keypt_idx in range(keypts_3d.shape[0]):
            (x, y, _) = project(keypts_3d[keypt_idx].reshape((3, 1)), R, T)
            keypts_2d[example_id, keypt_idx, 0] = x
            keypts_2d[example_id, keypt_idx, 1] = y
    # np.save(os.path.join('my_labels', args.object_name, 'keypoints_2d.npy'), keypts_2d)

if __name__ == '__main__':
    main()
