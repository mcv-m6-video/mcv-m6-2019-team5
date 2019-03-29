import cv2
import numpy as np


class BlockMatching:

    def __init__(self, block_size=8, window_size=64, stride=4, criteria="SAD"):
        self.block_size = block_size
        self.window_size = window_size
        self.stride = stride

        criteria_refs = {
            'SAD': self._sum_absolute_differences,
            'SSD': self._sum_square_differences
        }

        self.criteria = criteria_refs.get(criteria)

    def __call__(self, im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
        out = np.zeros((im1.shape[0], im1.shape[1], 2))
        self.img_shape = im1.shape
        # No padding used, if needed, we could use np.pad
        for i in range(self.window_size // 2, self.img_shape[0] - self.window_size // 2):
            for j in range(self.window_size // 2, self.img_shape[1] - self.window_size // 2):
                block = im1[i - self.block_size // 2:i + self.block_size // 2,
                            j - self.block_size // 2:j + self.block_size // 2, :]
                maximum_matching = self._find_maximum_matching(block, im2, (i, j))

                out[i, j, :] = maximum_matching

        return out

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
                        likelihood = self.criteria(box1, box2)
                        if likelihood > likelihood_aux:
                            likelihood_aux = likelihood
                            out = pixel2
        return out

    @staticmethod
    def _sum_absolute_differences(box1: np.ndarray, box2: np.ndarray) -> float:
        return float(np.sum(np.abs(box1 - box2)))

    @staticmethod
    def _sum_square_differences(box1: np.ndarray, box2: np.ndarray) -> float:
        return float(np.sum(np.power(box1 - box2, 2)))


if __name__ == '__main__':
    block = BlockMatching()
    im11 = cv2.imread("../../datasets/optical_flow/img/000045_10.png")
    im21 = cv2.imread("../../datasets/optical_flow/img/000045_11.png")
    im11 = cv2.cvtColor(im11, cv2.COLOR_BGR2GRAY)
    im21 = cv2.cvtColor(im21, cv2.COLOR_BGR2GRAY)

    block(im11, im21)
