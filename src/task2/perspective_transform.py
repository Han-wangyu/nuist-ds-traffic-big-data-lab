import cv2
import numpy as np

def order_points(pts):
    """
    对四个点进行排序，使其符合左上、右上、右下、左下的顺序
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # 计算左上角和右下角点
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上角点的x+y最小
    rect[2] = pts[np.argmax(s)]  # 右下角点的x+y最大
    
    # 计算右上角和左下角点
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上角点的x-y最小
    rect[3] = pts[np.argmax(diff)]  # 左下角点的x-y最大
    
    return rect

def perspective_transform(image, points):
    """
    对图像进行透视变换
    params:
        image: 原始图像
        points: 四个角点坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    returns:
        warped: 矫正后的图像
    """
    # 将点转换为numpy数组并排序
    pts = np.array(points, dtype="float32")
    rect = order_points(pts)
    
    # 设定目标图像的大小（与任务一保持一致）
    dst_width = 140
    dst_height = 44
    
    # 构建目标点坐标
    dst = np.array([
        [0, 0],
        [dst_width - 1, 0],
        [dst_width - 1, dst_height - 1],
        [0, dst_height - 1]
    ], dtype="float32")
    
    # 计算透视变换矩阵并应用
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (dst_width, dst_height))
    
    return warped 