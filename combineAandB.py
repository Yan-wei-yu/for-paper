from pdb import set_trace as st
import os
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='D:/Users/user/Desktop/weiyundontdelete/GANdata/forjournal/inlayonlay/train/downup')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='D:/Users/user/Desktop/weiyundontdelete/GANdata/forjournal/inlayonlay/train/gapprep')
parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default='D:/Users/user/Desktop/weiyundontdelete/GANdata/forjournal/inlayonlay/train/final')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images',type=int, default=1000)
parser.add_argument('--use_AB', dest='use_AB', help='if true: (0001_A, 0001_B) to (0001_AB)',action='store_true')
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg,  getattr(args, arg))

img_list_A = filter(lambda f: not f.startswith('.'), os.listdir(args.fold_A))  # ignore hidden folders like .DS_Store
img_list_B = filter(lambda f: not f.startswith('.'), os.listdir(args.fold_B))  # ignore hidden folders like .DS_Store

img_list = set(img_list_A).intersection(img_list_B)
num_imgs = min(args.num_imgs, len(img_list))

print(f'Use {num_imgs}/{len(img_list)} images for merging.')

img_fold_AB = args.fold_AB
if not os.path.isdir(img_fold_AB):
    os.makedirs(img_fold_AB)

for img_name in img_list:
    path_A = os.path.join(args.fold_A, img_name)
    path_B = os.path.join(args.fold_B, img_name)

    if os.path.isfile(path_A) and os.path.isfile(path_B):
        name_AB = img_name
        if args.use_AB:
            name_AB = name_AB.replace('_A.', '.')  # remove _A
        path_AB = os.path.join(img_fold_AB, name_AB)
        im_A = cv2.imread(path_A, cv2.IMREAD_UNCHANGED)
        # im_A_gray = cv2.cvtColor(im_A, cv2.COLOR_BGR2GRAY)#牙溝
        im_B = cv2.imread(path_B, cv2.IMREAD_UNCHANGED)
        im_AB = np.concatenate([im_A, im_B], 1)
        cv2.imwrite(path_AB, im_AB)
    else:
        print(f"Image name mismatch for {img_name} in {args.fold_A} and {args.fold_B}")

         