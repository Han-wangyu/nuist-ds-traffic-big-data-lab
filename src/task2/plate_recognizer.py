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
        加载任务一训练好的模型
        """
        try:
            self.province_clf = joblib.load(province_model_path)
            self.char_clf = joblib.load(char_model_path)
            print("模型加载成功")
        except Exception as e:
            raise Exception(f"模型加载失败: {str(e)}")
        
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
    
    def recognize(self, plate_img):
        """
        识别车牌
        params:
            plate_img: 矫正后的车牌图像
        returns:
            plate_number: 识别的车牌号码，如"皖A12345"
        """
        try:
            # 分割字符
            char_imgs = segment_characters(plate_img)
            if len(char_imgs) != 7:
                print(f"分割出 {len(char_imgs)} 个字符，期望7个")
                return None
            
            # 分别提取省份和字符特征
            province_features = extract_features_batch([char_imgs[0]])
            char_features = extract_features_batch(char_imgs[1:])
            
            # 预测
            province_pred = self.province_clf.predict(province_features)[0]
            char_preds = self.char_clf.predict(char_features)
            
            # 转换为车牌号
            province_char = self.province_map[province_pred]
            plate_chars = [self.char_map[pred] for pred in char_preds]
            plate_number = province_char + ''.join(plate_chars)
            
            return plate_number
            
        except Exception as e:
            print(f"识别过程出错: {str(e)}")
            return None 