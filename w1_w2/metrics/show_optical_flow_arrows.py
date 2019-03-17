import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def show_optical_flow_arrows(frame: np.ndarray, optical_flow: np.ndarray):
    _, ax = plt.subplots()

    ax.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    _flow_to_arrows(optical_flow[:, :, 0:2], ax)

    plt.axis('off')

    plt.show()


def _flow_to_arrows(flow_uv, ax):
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'

    hor_size = flow_uv.shape[0]
    vert_size = flow_uv.shape[1]

    u = np.zeros(flow_uv.shape[0:2])
    v = np.zeros(flow_uv.shape[0:2])

    step = 16

    for i in range(0, hor_size, step):
        for j in range(0, vert_size, step):
            block = flow_uv[i:(i + step), j:(j + step), :]
            module = pow(block[:, :, 0], 2) + pow(block[:, :, 1], 2)
            inx, iny = np.where(module == np.amax(module))

            u[i, j] = block[inx[0], iny[0], 0]
            v[i, j] = block[inx[0], iny[0], 1]

    print(u.shape)

    max_u_value = np.amax(u)
    max_v_value = np.amax(v)
    max_uv_value = max(max_u_value, max_v_value)

    x = [[i for i in range(0, vert_size, step)] for _ in range(0, hor_size, step)]
    y = [[i for _ in range(0, vert_size, step)] for i in range(0, hor_size, step)]

    ax.quiver(x, y, u[::step, ::step], v[::step, ::step], width=0.002, scale=max_uv_value * 20, color=(0, 0.1, 0.8))
