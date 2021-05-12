import numpy as np


def Prewitt():
    W_h = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) / 3
    return W_h, np.rot90(W_h)


def Sobel():
    W_h = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 4
    return W_h, np.rot90(W_h)


def Roberts():
    W_h = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])
    W_v = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    return W_h, W_v


def Isotropic_Sobel():
    W_h = np.array([[-1, 0, 1], [-2**0.5, 0, 2**0.5], [-1, 0, 1]]) / (2 + 2**0.5)
    return W_h, np.rot90(W_h)


def Prewitt_dirction():
    W_e = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) / 3
    W_en = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]]) / 3
    W_s = np.rot90(W_e)
    W_es = np.rot90(W_en)
    return W_e, W_en, -W_s, -W_es, -W_e, -W_en, -W_s, -W_es


def Sobel_dirction():
    W_e = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 4
    W_en = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]) / 4
    W_s = np.rot90(W_e)
    W_es = np.rot90(W_en)
    return W_e, W_en, -W_s, -W_es, -W_e, -W_en, -W_s, -W_es


def Kirsch_dirction():
    W_e  = np.array([[-3, -3,  5], [-3, 0,  5], [-3, -3,  5]]) / 15
    W_en = np.array([[-3,  5,  5], [-3, 0,  5], [-3, -3, -3]]) / 15
    W_s = np.rot90(W_e)
    W_w = np.rot90(W_s)
    W_n = np.rot90(W_w)
    W_es = np.rot90(W_en)
    W_ws = np.rot90(W_es)
    W_wn = np.rot90(W_ws)
    return W_e, W_en, W_n, W_wn, W_w, W_ws, W_s, W_es


def gradient_operator():
    W_h = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
    W_v = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
    return W_h, W_v


def line_detection_operator():
    W0 = np.array([[-1, -1, -1], [ 2, 2,  2], [-1, -1, -1]]) / 6
    W1 = np.array([[-1, -1,  2], [-1, 2, -1], [ 2, -1, -1]]) / 6
    return W0, W1, W0.T, np.rot90(W1)


def laplacian_operator(mode=4):
    assert mode in (4, 8)
    if mode == 4:
        return np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),
    else:
        return np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),


def LoG(sigma, size=3):
    r, c = np.mgrid[0:size:1, 0:size:1]
    r -= int((size-1)/2)
    c -= int((size-1)/2)
    sigma2 = sigma ** 2
    sigma4 = sigma ** 4
    norm2 = r ** 2 + c ** 2
    kernel = (norm2 / sigma4 - 2/sigma2) * np.exp(- norm2 / (2 * sigma2))
    return kernel - np.mean(kernel),


def Nevatia_Babu():
    w0 = np.ones((5, 5))
    w0 *= np.array([-100, -100, 0, 100, 100]) / 1000
    w270 = np.rot90(w0)
    w180 = np.rot90(w270)
    w90 = np.rot90(w180)
    w60 = np.array([[100] * 5, [-32, 78, 100, 100, 100],
                    [-100, -92, 0, 92, 100],
                    [-100, -100, -100, -78, 32],
                    [-100] * 5]) / 1102
    w330 = np.rot90(w60)
    w240 = np.rot90(w330)
    w150 = np.rot90(w240)
    w30 = -w60.T
    w300 = np.rot90(w30)
    w210 = np.rot90(w300)
    w120 = np.rot90(w210)
    return w0, w30, w60, w90, w120, w150, w180, w210, w240, w270, w300, w330

