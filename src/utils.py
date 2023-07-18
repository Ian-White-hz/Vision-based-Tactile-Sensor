import math

import cv2
import numpy as np
from pynhhd import nHHD


def put_optical_flow_arrows_on_image(image, optical_flow, threshold=4.0):
    # Don't affect original image
    image = image.copy()

    scaled_flow = optical_flow * 2.0  # scale factor

    # Get start and end coordinates of the optical flow
    flow_start = np.stack(
        np.meshgrid(range(0, scaled_flow.shape[1], 30),
                    range(0, scaled_flow.shape[0], 30)), 2)
    flow_end = (scaled_flow[flow_start[:, :, 1], flow_start[:, :, 0], :] +
                flow_start).astype(np.int32)

    # Threshold values
    norm = np.linalg.norm(scaled_flow[flow_start[:, :, 1], flow_start[:, :,
                                                                      0], :],
                          axis=2)
    # print(norm.max(), norm.min())
    norm[norm < threshold] = 0
    # Draw all the nonzero values
    nz = np.nonzero(norm)

    norm = np.asarray(norm / 50.0 * 255.0, dtype='uint8')
    color_image = cv2.applyColorMap(norm, cv2.COLORMAP_RAINBOW).astype('int')
    for i in range(len(nz[0])):
        y, x = nz[0][i], nz[1][i]
        cv2.arrowedLine(image,
                        pt1=tuple(flow_start[y, x]),
                        pt2=tuple(flow_end[y, x]),
                        color=(int(color_image[y, x,
                                               0]), int(color_image[y, x, 1]),
                               int(color_image[y, x, 2])),
                        thickness=1,
                        tipLength=.3)
    return image

def show_multi_imgs(scale, imglist, order=None, border=5, border_color=(255, 255, 0)):
    """
    :param scale: float 原图缩放的尺度
    :param imglist: list 待显示的图像序列
    :param order: list or tuple 显示顺序 行×列
    :param border: int 图像间隔距离
    :param border_color: tuple 间隔区域颜色
    :return: 返回拼接好的numpy数组
    """
    if order is None:
        order = [1, len(imglist)]
    allimgs = imglist.copy()
    ws , hs = [], []
    for i, img in enumerate(allimgs):
        if np.ndim(img) == 2:
            allimgs[i] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        allimgs[i] = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale)
        ws.append(allimgs[i].shape[1])
        hs.append(allimgs[i].shape[0])
    w = max(ws)
    h = max(hs)
    # 将待显示图片拼接起来
    sub = int(order[0] * order[1] - len(imglist))
    # 判断输入的显示格式与待显示图像数量的大小关系
    if sub > 0:
        for s in range(sub):
            allimgs.append(np.zeros_like(allimgs[0]))
    elif sub < 0:
        allimgs = allimgs[:sub]
    imgblank = np.zeros(((h+border) * order[0], (w+border) * order[1], 3)) + border_color
    imgblank = imgblank.astype(np.uint8)
    for i in range(order[0]):
        for j in range(order[1]):
            imgblank[(i * h + i*border):((i + 1) * h+i*border), (j * w + j*border):((j + 1) * w + j*border), :] = allimgs[i * order[1] + j]
    return imgblank

def flowvis(frame, flow, type=1):
    hsv = np.zeros_like(frame)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    if type == 1:
        hsv[...,1] = 255
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    elif type == 2:
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = 255
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return bgr

def Helmholtz(flow):
    dims = flow.shape
    dims = dims[0:2]
    dx = (0.01, 0.01) 
    Y, X = np.mgrid[0:dims[0], 0:dims[1]]

    mvf = np.linalg.norm(flow, axis=2)

    #vrng = (0, 7.07106781187)
    vrng = (0, mvf.max())

    nhhd = nHHD(grid=dims, spacings=dx)
    nhhd.decompose(flow)

    r = nhhd.r
    d = nhhd.d
    h = nhhd.h

    Sn = np.linalg.norm(d, ord=1, axis=2)
    Sn = Sn.sum()/(dims[0]*dims[1])

    Sr = flow.sum(axis=0).sum(axis=0)
    St = ((Sr[0]/dims[0])**2+(Sr[1]/dims[1])**2)**(1/2)

    return Sn,St,d


def getContactBoundary(density, kernel=np.ones((5, 5), np.uint8)):
    threshold, contact_area = cv2.threshold(density,
                                            0,
                                            2,
                                            type=cv2.THRESH_BINARY +
                                            cv2.THRESH_OTSU)

    if threshold > 5:
        contact_area = cv2.morphologyEx(contact_area,
                                        cv2.MORPH_OPEN,
                                        kernel,
                                        iterations=3)
    else:
        contact_area = np.zeros_like(contact_area)
    contact_boundary = np.gradient(contact_area.astype('float32'))
    contact_boundary = np.abs(contact_boundary[0].astype('uint8')) + np.abs(
        contact_boundary[1].astype('uint8'))
    _, contact_boundary = cv2.threshold(contact_boundary,
                                        0,
                                        255,
                                        type=cv2.THRESH_BINARY)
    # lines = cv2.HoughLines(contact_boundary, 8, np.pi / 20, 30) # line detection in boundaries
    contact_boundary = cv2.cvtColor(contact_boundary, cv2.COLOR_GRAY2BGR)
    # if lines is not None and len(lines) > 1:
    #     lines = lines[:2]
    #     if np.abs(lines[0][0][1] - lines[1][0][1]) < np.pi / 4.0:
    #         if np.abs(lines[0][0][1] - lines[1][0][1]) < 1e-2:
    #             a = -1
    #             b = -1
    #         else:
    #             Theta = np.array(
    #                 [[math.cos(lines[0][0][1]),
    #                   math.sin(lines[0][0][1])],
    #                  [math.cos(lines[1][0][1]),
    #                   math.sin(lines[1][0][1])]])
    #             Rho = np.array([[lines[0][0][0]], [lines[1][0][0]]])
    #             intersection = np.matmul(np.linalg.inv(Theta), Rho)
    #             a = intersection[0].item()
    #             b = intersection[1].item()
    #         if not (a > 0 and a < density.shape[1] and b > 0
    #                 and b < density.shape[0]):
    #             for i in range(0, len(lines)):
    #                 rho = lines[i][0][0]
    #                 theta = lines[i][0][1]
    #                 a = math.cos(theta)
    #                 b = math.sin(theta)
    #                 x0 = a * rho
    #                 y0 = b * rho
    #                 pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    #                 pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
    #                 cv2.line(contact_boundary, pt1, pt2, (0, 0, 255), 3,
    #                          cv2.LINE_AA)
    return threshold, contact_boundary
