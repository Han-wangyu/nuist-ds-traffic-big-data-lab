import cv2
import numpy as np
import pandas as pd
import os
from perspective_transform import perspective_transform
from plate_recognizer import PlateRecognizer

def load_test_data(csv_path, image_dir):
    """
    加载测试数据
    """
    df = pd.read_csv(csv_path)
    results = []
    
    print(f"正在读取CSV文件: {csv_path}")
    print(f"图像目录: {image_dir}")
    
    for idx, row in df.iterrows():
        try:
            # 读取原始图像
            img_path = os.path.join(image_dir, f"{int(row['filename'])}.jpg")
            img = cv2.imread(img_path)
            if img is None:
                print(f"无法读取图片: {img_path}")
                continue
            
            # 获取四个角点坐标
            points = [
                [row['lux'], row['luy']],  # 左上
                [row['rux'], row['ruy']],  # 右上
                [row['rdx'], row['rdy']],  # 右下
                [row['ldx'], row['ldy']]   # 左下
            ]
            
            # 获取真实标签
            label = [
                int(row['prov']),
                int(row['str_num1']),
                int(row['str_num2']),
                int(row['str_num3']),
                int(row['str_num4']),
                int(row['str_num5']),
                int(row['str_num6'])
            ]
            
            results.append((img, points, label))
            
        except Exception as e:
            print(f"处理第 {idx} 行时出错: {str(e)}")
    
    print(f"成功加载 {len(results)} 张图片")
    return results

def main():
    # 配置路径
    test_csv = "data/test.csv"
    test_image_dir = "data/test/original"  # 使用未矫正的原始图像
    model_dir = "models"
    
    # 加载识别器
    recognizer = PlateRecognizer(
        os.path.join(model_dir, "province_model.pkl"),
        os.path.join(model_dir, "char_model.pkl")
    )
    
    # 加载测试数据
    test_data = load_test_data(test_csv, test_image_dir)
    
    correct_count = 0
    total_count = 0
    
    # 处理每张图片
    for idx, (img, points, true_label) in enumerate(test_data):
        try:
            # 1. 透视变换
            warped = perspective_transform(img, points)
            
            # 2. 识别车牌
            pred_plate = recognizer.recognize(warped)
            if pred_plate is None:
                print(f"图片 {idx}: 识别失败")
                continue
            
            # 3. 获取真实车牌号
            true_province = recognizer.province_map[true_label[0]]
            true_chars = [recognizer.char_map[x] for x in true_label[1:]]
            true_plate = true_province + ''.join(true_chars)
            
            # 4. 比较结果
            is_correct = (pred_plate == true_plate)
            if is_correct:
                correct_count += 1
            total_count += 1
            
            # 5. 输出结果
            print(f"\n图片 {idx}:")
            print(f"预测结果: {pred_plate}")
            print(f"真实标签: {true_plate}")
            print(f"是否正确: {'✓' if is_correct else '✗'}")
            
        except Exception as e:
            print(f"处理图片 {idx} 时出错: {str(e)}")
            continue
        
        # 每处理10张图片输出一次当前准确率
        if (idx + 1) % 10 == 0:
            current_acc = correct_count / total_count if total_count > 0 else 0
            print(f"\n当前进度: {idx + 1}/{len(test_data)}")
            print(f"当前准确率: {current_acc:.4f}")
    
    # 输出最终结果
    final_accuracy = correct_count / total_count if total_count > 0 else 0
    print("\n最终结果:")
    print(f"总样本数: {total_count}")
    print(f"正确识别: {correct_count}")
    print(f"识别准确率: {final_accuracy:.4f}")

if __name__ == "__main__":
    main() 