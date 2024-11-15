import cv2
import numpy as np

def segment_characters(plate_img):
    """
    分割车牌字符
    params:
        plate_img: 预处理后的车牌图像（可以是灰度图或彩色图）
    returns:
        字符图像列表 [省份字符, 字母数字字符...]
    """
    # 确保输入是灰度图
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img
    
    # 二值化处理
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 定义字符大致位置（基于固定比例）
    height, width = binary.shape
    char_positions = [
        (0, int(width * 0.15)),          # 省份字符
        (int(width * 0.15), int(width * 0.3)),  # 第一个字符
        (int(width * 0.3), int(width * 0.45)),  # 第二个字符
        (int(width * 0.45), int(width * 0.6)),  # 第三个字符
        (int(width * 0.6), int(width * 0.75)),  # 第四个字符
        (int(width * 0.75), int(width * 0.9)),  # 第五个字符
        (int(width * 0.9), width)        # 第六个字符
    ]
    
    char_imgs = []
    for start, end in char_positions:
        # 提取字符区域
        char_region = binary[:, start:end]
        # 统一大小
        char_img = cv2.resize(char_region, (20, 40))
        char_imgs.append(char_img)
    
    return char_imgs 