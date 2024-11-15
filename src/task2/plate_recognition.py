import cv2
import numpy as np
import joblib
import sys
import os

# 添加父目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task1.char_segmentation import segment_characters
from task1.feature_extraction import extract_features_batch

class PlateRecognizer:
    def __init__(self, province_model_path, char_model_path):
        """
        加载训练好的模型
        """
        self.province_clf = joblib.load(province_model_path)
        self.char_clf = joblib.load(char_model_path)
        
        # 字符映射（与task1保持一致）
        self.province_map = {
            0: "皖", 1: "沪", 2: "津", 3: "渝", 4: "冀", 5: "晋", 6: "蒙",
            7: "辽", 8: "吉", 9: "黑", 10: "苏", 11: "浙", 12: "京",
            13: "闽", 14: "赣", 15: "鲁", 16: "豫", 17: "鄂", 18: "湘",
            19: "粤", 20: "桂", 21: "琼", 22: "川", 23: "贵", 24: "云",
            25: "藏", 26: "陕", 27: "甘", 28: "青", 29: "宁", 30: "新"
        }
        
        self.char_map = {
            0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H",
            8: "J", 9: "K", 10: "L", 11: "M", 12: "N", 13: "P", 14: "Q",
            15: "R", 16: "S", 17: "T", 18: "U", 19: "V", 20: "W", 21: "X",
            22: "Y", 23: "Z", 24: "0", 25: "1", 26: "2", 27: "3", 28: "4",
            29: "5", 30: "6", 31: "7", 32: "8", 33: "9"
        }
    
    def preprocess_image(self, img):
        """
        预处理图像
        """
        # 转换为灰度图
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # 自适应二值化
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        
        # 去噪
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    def recognize(self, plate_img):
        """
        识别车牌
        """
        try:
            # 预处理图像
            processed_img = self.preprocess_image(plate_img)
            
            # 分割字符
            char_imgs = segment_characters(processed_img)
            if len(char_imgs) != 7:
                print(f"分割出 {len(char_imgs)} 个字符，期望7个")
                return None
            
            # 分别处理省份字符和其他字符
            province_img = char_imgs[0]
            other_chars = char_imgs[1:]
            
            # 分别提取特征
            province_features = extract_features_batch([province_img])  # 单独处理省份
            char_features = extract_features_batch(other_chars)        # 处理其他字符
            
            # 分别预测
            try:
                # 打印特征形状用于调试
                print(f"省份特征形状: {province_features.shape}")
                print(f"字符特征形状: {char_features.shape}")
                
                province_pred = self.province_clf.predict(province_features)[0]
                char_preds = self.char_clf.predict(char_features)
                
                # 打印原始预测值用于调试
                print(f"省份原始预测值: {province_pred}")
                print(f"字符原始预测值: {char_preds}")
                
                # 转换为字符串
                province_char = self.province_map[province_pred]
                plate_chars = [self.char_map[pred] for pred in char_preds]
                
                # 调试信息
                print("预测结果：")
                print(f"省份预测值: {province_pred} -> {province_char}")
                print(f"字符预测值: {char_preds} -> {''.join(plate_chars)}")
                
                return province_char + ''.join(plate_chars)
                
            except Exception as e:
                print(f"预测过程出错: {str(e)}")
                print(f"省份特征形状: {province_features.shape}")
                print(f"字符特征形状: {char_features.shape}")
                return None
                
        except Exception as e:
            print(f"识别过程出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None 