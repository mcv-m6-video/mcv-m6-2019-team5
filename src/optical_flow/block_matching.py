import numpy as np
import cv2


class BlockMatching:

    def __init__(self, block_size=8, window_size=64, stride=4, criteria="SAD"):
        self.block_side = block_size
        self.window_size = window_size
        self.stride = stride

        criteria_refs = {
            'SAD': self._sum_absolut_diferences,
            'SSD': self._sum_square_diferences
        }

        self.criteria = criteria_refs.get(criteria)

    def __call__(self, im1: np.ndarray, im2: np.ndarray, ) -> np.ndarray:
        self.img_shape = im1.shape
        for i in np.ndindex(im1.shape[:2]):
            box1 = self._get_block(im1, i)
            maximum_matching = self._find_maximum_matching(box1, im2, i)

    def _find_maximum_matching(self, box1: np.ndarray, im2: np.ndarray, pixel1: tuple) -> tuple:
        likelihood_aux = 0
        out = (0, 0)
        for col in range(-int(self.window_size / 2), int(self.window_size / 2) + 1, self.stride):
            col_index = col + pixel1[0]
            if 0 <= col_index < self.img_shape[0]:
                for row in range(-int(self.window_size / 2), int(self.window_size / 2) + 1, self.stride):
                    row_index = row + pixel1[1]
                    if 0 <= row_index < self.img_shape[1]:
                        pixel2 = (col_index, row_index)
                        box2 = self._get_block(im2, pixel2)
                        likelihood = self._sum_absolut_diferences(box1, box2)
                        if likelihood > likelihood_aux:
                            likelihood_aux = likelihood
                            out = pixel2
        return out

    def _get_block(self, im: np.ndarray, pixel: tuple) -> np.ndarray:
        out = np.zeros((self.block_side, self.block_side))
        col_index_out = 0
        for col in range(-int(self.block_side / 2), int(self.block_side / 2) + 1, self.stride):
            col_index = col + pixel[0]
            row_index_out = 0
            if 0 <= col_index < self.img_shape[0]:
                for row in range(-int(self.block_side / 2), int(self.block_side / 2) + 1, self.stride):
                    row_index = row + pixel[1]
                    if 0 <= row_index < self.img_shape[1]:
                        out[col_index_out, row_index_out] = im[col_index, row_index]
                    row_index_out += 1
                col_index_out += 1
        return out

    def _sum_absolut_diferences(self, box1: np.ndarray, box2: np.ndarray):
        pass

    def _sum_square_diferences(self, box1: np.ndarray, box2: np.ndarray):
        pass


block = BlockMatching()
im11 = cv2.imread("../../datasets/optical_flow/img/000045_10.png")
im21 = cv2.imread("../../datasets/optical_flow/img/000045_11.png")
im11 = cv2.cvtColor(im11, cv2.COLOR_BGR2GRAY)
im21 = cv2.cvtColor(im21, cv2.COLOR_BGR2GRAY)

block(im11, im21)
