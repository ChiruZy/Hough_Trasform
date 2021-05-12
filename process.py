import numpy as np
import matplotlib.pyplot as plt


def conv_process(img, operators, k_size):
    space = k_size // 2
    raw = np.pad(img, [space, space], 'edge')
    w, h = raw.shape

    res = [np.zeros(img.shape) for _ in range(len(operators))]
    for row in range(space, w - space):
        for col in range(space, h - space):
            for g, o in zip(res, operators):
                block = raw[row - space:row + space + 1, col - space:col + space + 1]
                g[row - space, col - space] = np.sum(block * o)
    return res


def get_edge_gradient(gs):
    gs = np.abs(np.array(gs))
    return gs.max(axis=0)


def threshold_seg(g, thres):
    g = (g - np.min(g)) / (np.max(g) - np.min(g)) * 255
    return np.where(g > thres, 255, 0).astype('uint8')


def plot_imgs(row, col, imgs, titles):
    for idx, (img, title) in enumerate(zip(imgs, titles)):
        plt.subplot(row, col, idx + 1)
        plt.title(title)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.show()