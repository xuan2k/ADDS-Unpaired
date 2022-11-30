from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset_unpaired import MonoDatasetUnpaired
from PIL import Image


class KITTIDatasetUnpaired(MonoDatasetUnpaired):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDatasetUnpaired, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size    
        # self.K = np.array([[0.58, 0, 0.5, 0],
        #                    [0, 1.92, 0.5, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1]], dtype=np.float32)

        self.K = np.array([[0.768, 0, 0.5, 0],
                           [0, 1.024, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        # 983.044006, 0, 643.646973 / 1280
        # 0, 983.044006, 493.378998 / 960
        self.full_res_shape = (1280, 640)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        # color = color.crop((0, 160, 1280, 960-160))
        # color = color.resize((512, 256),Image.ANTIALIAS)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class KITTIRAWDatasetUnpaired(KITTIDatasetUnpaired):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDatasetUnpaired, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        # calib_path = os.path.join(self.data_path, folder.split("/")[0])
        #
        # velo_filename = os.path.join(
        #     self.data_path,
        #     folder,
        #     "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        depth_path = os.path.join(self.data_path, folder+'_gt', f_str)

        img_file = Image.open(depth_path)
        depth_png = np.array(img_file, dtype=int)
        img_file.close()
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert np.max(depth_png) > 255, \
            "np.max(depth_png)={}, path={}".format(np.max(depth_png), depth_path)
        # print(np.min(depth_png), ' ',np.max(depth_png))

        depth_gt = depth_png.astype(np.float) / 256.
        # depth[depth_png == 0] = -1.
        # depth = np.expand_dims(depth, -1)

        # depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])

        depth_gt = depth_gt[160:960-160,:]

        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt