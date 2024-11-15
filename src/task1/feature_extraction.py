import numpy as np
from sklearn.preprocessing import StandardScaler
import cv2

def extract_hog_features(char_img):
    """
    提取HOG特征
    """
    # 确保图像大小统一
    img_resized = cv2.resize(char_img, (20, 40))
    
    # 计算HOG特征
    winSize = (20, 40)
    blockSize = (10, 10)
    blockStride = (5, 5)
    cellSize = (5, 5)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    hog_features = hog.compute(img_resized)
    
    return hog_features.flatten()

def extract_projection_features(char_img):
    """
    提取水平和垂直投影特征
    """
    # 水平投影
    h_proj = np.sum(char_img, axis=1)
    # 垂直投影
    v_proj = np.sum(char_img, axis=0)
    
    # 归一化
    h_proj = h_proj / np.max(h_proj) if np.max(h_proj) > 0 else h_proj
    v_proj = v_proj / np.max(v_proj) if np.max(v_proj) > 0 else v_proj
    
    return np.concatenate([h_proj, v_proj])

def extract_contour_features(char_img):
    """
    提取轮廓特征
    """
    # 二值化
    _, binary = cv2.threshold(char_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return np.zeros(20)
    
    # 获取最大轮廓
    max_contour = max(contours, key=cv2.contourArea)
    
    # 计算轮廓特征
    area = cv2.contourArea(max_contour)
    perimeter = cv2.arcLength(max_contour, True)
    x, y, w, h = cv2.boundingRect(max_contour)
    aspect_ratio = float(w)/h if h > 0 else 0
    
    # 计算Hu矩
    moments = cv2.moments(max_contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # 组合特征
    features = [area, perimeter, aspect_ratio]
    features.extend(hu_moments)
    
    # 填充到固定长度
    features = np.pad(features, (0, 20-len(features)), 'constant')
    
    return features

def extract_distribution_features(char_img):
    """
    提取像素分布特征
    """
    h, w = char_img.shape
    cells = (4, 2)  # 将图像分成8个区域
    cell_h, cell_w = h // cells[0], w // cells[1]
    
    features = []
    for i in range(cells[0]):
        for j in range(cells[1]):
            cell = char_img[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            # 计算区域的平均灰度值和标准差
            mean = np.mean(cell)
            std = np.std(cell)
            features.extend([mean, std])
    
    return np.array(features)

def extract_features_batch(char_imgs):
    """
    批量提取特征，并进行统一标准化
    params:
        char_imgs: 字符图像列表，每个元素是一个二维灰度图像
    returns:
        normalized_features: 标准化后的特征矩阵
    """
    # 确保输入是列表
    if not isinstance(char_imgs, list):
        char_imgs = [char_imgs]
    
    # 提取原始特征
    raw_features = []
    for img in char_imgs:
        try:
            # 确保图像是2D的
            if len(img.shape) != 2:
                print(f"警告：图像维度不正确: {img.shape}")
                continue
                
            # HOG特征
            hog_features = extract_hog_features(img)
            # 投影特征
            projection_features = extract_projection_features(img)
            # 轮廓特征
            contour_features = extract_contour_features(img)
            # 像素分布特征
            distribution_features = extract_distribution_features(img)
            
            # 合并该图像的所有特征
            combined = np.concatenate([hog_features, projection_features, 
                                     contour_features, distribution_features])
            raw_features.append(combined)
            
        except Exception as e:
            print(f"特征提取错误: {str(e)}")
            continue
    
    if not raw_features:
        raise ValueError("没有成功提取到任何特征！")
    
    # 将所有特征转换为数组并一起标准化
    features_array = np.array(raw_features)
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features_array)
    
    return normalized_features

def build_dataset(char_imgs, labels):
    """
    构建训练数据集
    params:
        char_imgs: 字符图像列表
        labels: 对应的标签列表
    returns:
        X: 特征矩阵
        y: 标签数组
    """
    if len(char_imgs) != len(labels):
        raise ValueError(f"图像数量({len(char_imgs)})和标签数量({len(labels)})不匹配！")
    
    # 提取特征
    try:
        X = extract_features_batch(char_imgs)
        y = np.array(labels)
        
        print(f"特征矩阵形状: {X.shape}")
        print(f"标签数组形状: {y.shape}")
        
        return X, y
    except Exception as e:
        print(f"构建数据集失败: {str(e)}")
        raise 