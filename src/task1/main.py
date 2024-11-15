import os
import cv2
import numpy as np
import pandas as pd
from char_segmentation import segment_characters
from feature_extraction import build_dataset
from classifier import PlateClassifier

# 添加字符映射字典
PROVINCE_MAP = {
    "皖": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6,
    "辽": 7, "吉": 8, "黑": 9, "苏": 10, "浙": 11, "京": 12,
    "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18,
    "粤": 19, "桂": 20, "琼": 21, "川": 22, "贵": 23, "云": 24,
    "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30
}

CHAR_MAP = {
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7,
    "J": 8, "K": 9, "L": 10, "M": 11, "N": 12, "P": 13, "Q": 14,
    "R": 15, "S": 16, "T": 17, "U": 18, "V": 19, "W": 20, "X": 21,
    "Y": 22, "Z": 23, "0": 24, "1": 25, "2": 26, "3": 27, "4": 28,
    "5": 29, "6": 30, "7": 31, "8": 32, "9": 33
}

# 反向映射，用于将数字转换回字符
REVERSE_PROVINCE_MAP = {v: k for k, v in PROVINCE_MAP.items()}
REVERSE_CHAR_MAP = {v: k for k, v in CHAR_MAP.items()}

def convert_label(num, is_province=False):
    """
    将CSV中的标签转换为正确的值
    CSV中的数字已经是映射后的值，所以直接返回整数即可
    """
    try:
        value = int(float(num))
        # 验证标签值是否在有效范围内
        if is_province and value not in REVERSE_PROVINCE_MAP:
            print(f"无效的省份标签: {value}")
            return None
        elif not is_province and value not in REVERSE_CHAR_MAP:
            print(f"无效的字符标签: {value}")
            return None
        return value
    except Exception as e:
        print(f"标签转换错误: {num}, 错误类型: {str(e)}")
        return None

def load_data(csv_path, image_dir):
    """
    从CSV文件加载数据
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件 {csv_path} 不存在")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"图像目录 {image_dir} 不存在")
    
    df = pd.read_csv(csv_path)
    images = []
    labels = []
    
    print(f"正在读取CSV文件: {csv_path}")
    print(f"CSV文件包含 {len(df)} 条记录")
    
    # 打印一些示例数据用于调试
    print("\n数据示例（前3行）:")
    print(df.head(3))
    
    for idx, row in df.iterrows():
        try:
            img_path = os.path.join(image_dir, f"{int(row['filename'])}.jpg")
            
            if not os.path.exists(img_path):
                print(f"警告：找不到图片文件 {img_path}")
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告：无法读取图片 {img_path}")
                continue
            
            # 转换标签并验证
            label = [
                convert_label(row['prov'], is_province=True),
                convert_label(row['str_num1']),
                convert_label(row['str_num2']),
                convert_label(row['str_num3']),
                convert_label(row['str_num4']),
                convert_label(row['str_num5']),
                convert_label(row['str_num6'])
            ]
            
            # 打印一些标签用于调试
            if idx < 3:
                print(f"\n处理第 {idx} 行:")
                print(f"原始标签: {[row['prov'], row['str_num1'], row['str_num2'], row['str_num3'], row['str_num4'], row['str_num5'], row['str_num6']]}")
                print(f"转换后标签: {label}")
                if all(l is not None for l in label):
                    print("标签转换成功")
                    # 打印对应的字符（用于验证）
                    province_char = REVERSE_PROVINCE_MAP[label[0]]
                    chars = [REVERSE_CHAR_MAP[l] for l in label[1:]]
                    print(f"对应字符: {province_char} {''.join(chars)}")
            
            # 确保所有标签都有效
            if all(l is not None for l in label):
                images.append(img)
                labels.append(label)
            
        except Exception as e:
            print(f"处理第 {idx} 行时出错: {str(e)}")
            continue
        
        if idx % 100 == 0:
            print(f"已处理 {idx+1}/{len(df)} 条记录")
    
    print(f"\n成功加载 {len(images)} 张图片")
    print(f"标签示例（前3个）:")
    for i in range(min(3, len(labels))):
        print(f"图片 {i}: {labels[i]}")
    
    return images, labels

def evaluate_model(classifier, test_images, test_labels):
    """
    在测试集上评估模型
    """
    all_char_images = []
    province_labels = []
    char_labels = []
    
    for idx, (img, label) in enumerate(zip(test_images, test_labels)):
        char_images = segment_characters(img)
        
        if len(char_images) == 7:
            all_char_images.extend(char_images)
            province_labels.append(label[0])
            char_labels.extend(label[1:])
    
    # 构建测试集特征
    X_province_test, y_province_test = build_dataset(
        [all_char_images[i] for i in range(0, len(all_char_images), 7)],
        province_labels
    )
    X_char_test, y_char_test = build_dataset(
        [char for i, char in enumerate(all_char_images) if i % 7 != 0],
        char_labels
    )
    
    # 预测并计算准确率
    province_predictions = classifier.province_clf.predict(X_province_test)
    char_predictions = classifier.char_clf.predict(X_char_test)
    
    province_acc = np.mean(province_predictions == y_province_test)
    char_acc = np.mean(char_predictions == y_char_test)
    
    return province_acc, char_acc

def main():
    # 加载训练数据
    train_csv = "data/train.csv"
    test_csv = "data/test.csv"
    image_dir = "data/train/cropped"
    test_image_dir = "data/test/cropped"
    
    print("正在加载训练数据...")
    train_images, train_labels = load_data(train_csv, image_dir)
    if len(train_images) == 0:
        raise ValueError("没有成功加载任何训练图片！")
    
    print("正在加载测试数据...")
    test_images, test_labels = load_data(test_csv, test_image_dir)
    if len(test_images) == 0:
        raise ValueError("没有成功加载任何测试图片！")
    
    # 处理训练数据
    all_char_images = []
    province_labels = []
    char_labels = []
    
    print("正在分割字符...")
    for idx, (img, label) in enumerate(zip(train_images, train_labels)):
        char_images = segment_characters(img)
        
        if len(char_images) == 7:
            all_char_images.extend(char_images)
            province_labels.append(label[0])
            char_labels.extend(label[1:])
        else:
            print(f"警告：图片 {idx} 分割出 {len(char_images)} 个字符，期望7个")
    
    print(f"分割出 {len(all_char_images)} 个字符图像")
    print(f"省份标签数量: {len(province_labels)}")
    print(f"字符标签数量: {len(char_labels)}")
    
    if len(all_char_images) == 0:
        raise ValueError("字符分割失败！")
    
    # 构建训练特征矩阵
    print("正在提取特征...")
    try:
        # 分别处理省份字符和其他字符
        province_images = [all_char_images[i] for i in range(0, len(all_char_images), 7)]
        other_chars = [char for i, char in enumerate(all_char_images) if i % 7 != 0]
        
        print(f"省份字符数量: {len(province_images)}")
        print(f"其他字符数量: {len(other_chars)}")
        
        X_province, y_province = build_dataset(province_images, province_labels)
        X_char, y_char = build_dataset(other_chars, char_labels)
        
        print(f"省份特征矩阵形状: {X_province.shape}")
        print(f"字符特征矩阵形状: {X_char.shape}")
        
    except Exception as e:
        print(f"特征提取失败: {str(e)}")
        raise
    
    # 训练分类器
    print("开始训练分类器...")
    classifier = PlateClassifier()
    train_province_acc, train_char_acc = classifier.train(X_province, y_province, X_char, y_char)
    
    print("\n训练集结果：")
    print(f"省份识别准确率: {train_province_acc:.4f}")
    print(f"字符识别准确率: {train_char_acc:.4f}")
    
    # 在测试集上评估
    print("\n正在评估测试集...")
    test_province_acc, test_char_acc = evaluate_model(classifier, test_images, test_labels)
    
    print("\n测试集结果：")
    print(f"省份识别准确率: {test_province_acc:.4f}")
    print(f"字符识别准确率: {test_char_acc:.4f}")
    
    # 保存模型
    os.makedirs("models", exist_ok=True)
    classifier.save_models("models/province_model.pkl", "models/char_model.pkl")
    print("\n模型已保存到 models/ 目录")

if __name__ == "__main__":
    main() 