import matplotlib.pyplot as plt
from PIL import Image
from operators import Sobel
from process import conv_process, get_edge_gradient, threshold_seg, plot_imgs
import numpy as np


def hough_line_transform(binary_img):
    w, h = binary_img.shape
    max_len = int(round(np.sqrt(w ** 2 + h ** 2)))
    thetas = np.deg2rad(np.linspace(-180, 0, 2*max_len))
    rhos = np.linspace(-max_len, max_len, 2*max_len)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    n_thetas = len(thetas)
    accumulator = np.zeros((2*max_len, n_thetas), dtype=np.int16)
    y_idxs, x_idxs = np.nonzero(binary_img > 0)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(n_thetas):
            rho = max_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            accumulator[rho, t_idx] += 1
    return accumulator, thetas, rhos


def unpack_hough_line(hough_line, t, r, shape):
    hough_line = (hough_line - np.min(hough_line)) / (np.max(hough_line) - np.min(hough_line)) * 255
    ls = np.argsort(hough_line.flatten())[-3:]

    lines = np.zeros(shape, dtype=np.uint8)
    kbs = []
    for l in ls:
        ir, it = np.unravel_index(l, hough_line.shape)
        rho, theta = r[ir], t[it]
        k = -np.cos(theta) / np.sin(theta)
        b = rho / np.sin(theta)
        kbs.append((k, b))

    for i in range(len(kbs)):
        k0, b0 = kbs[i]
        cross = []
        for idx, (k, b) in enumerate(kbs):
            if idx == i:
                continue
            cross.append(int(round((b - b0) / (k0 - k))))
        line = np.array([int(round(k0 * x + b0)) for x in range(img.shape[0])])
        lines[line[min(cross):max(cross)], list(range(min(cross), max(cross)))] = 255
    return lines


def hough_circle_transform(binary_img, r_min, r_max):
    w, h = binary_img.shape

    max_len = int(round(np.sqrt(w ** 2 + h ** 2)))
    thetas = np.deg2rad(np.linspace(0, 360, 100))
    rhos = np.linspace(0, max_len, 100)
    radius = np.arange(r_min, r_max)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    n_thetas = len(thetas)
    n_radius = len(radius)

    accumulator = np.zeros((w, h, n_radius), dtype=np.int16)
    y_idxs, x_idxs = np.nonzero(binary_img > 0)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(n_thetas):
            for rho in rhos:
                a = int(round(x - rho * cos_t[t_idx]))
                b = int(round(y - rho * sin_t[t_idx]))
                r = int(round(np.sqrt((a - x) ** 2 + (b - y) ** 2)))
                if a < 0 or a >= w or b <0 or b>=h or r >= r_max or r < r_min:
                    continue
                accumulator[a, b, r-r_min] += 1
    return accumulator


def draw_circle(shape, center, r):
    img = np.zeros(shape)
    theta = np.linspace(0, 2 * np.pi, 360)
    a, b = center
    for t in theta:
        x = int(round(a + r * np.cos(t)))
        y = int(round(b + r * np.sin(t)))
        img[x, y] = 255
    return img


if __name__ == '__main__':
    img = np.array(Image.open('hough.png').convert('L'))
    sobel_img = get_edge_gradient(conv_process(img, Sobel(), 3))

    binary = threshold_seg(sobel_img, 140)
    hough_line, t, r = hough_line_transform(binary)
    lines = unpack_hough_line(hough_line, t, r, img.shape)

    binary = threshold_seg(sobel_img, 80)
    r_min, r_max = 10, 200
    hough_circle = hough_circle_transform(binary, r_min, r_max)
    x, y, r = np.unravel_index(np.argmax(hough_circle), hough_circle.shape)
    circle = draw_circle(img.shape, (x, y), r+r_min)

    hough = np.zeros(img.shape)
    hough[(circle > 0) | (lines > 0)] = 255

    plot_imgs(2, 4, [img, sobel_img, hough_line, lines, hough_circle.sum(axis=-1), circle, hough],
              ['raw', 'sobel', 'hough line transform', 'line detection', 'hough circle transform', 'circle detection',
               'hough detection'])
    plt.show()

