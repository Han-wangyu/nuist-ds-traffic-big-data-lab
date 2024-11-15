from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib
from sklearn.ensemble import RandomForestClassifier

class PlateClassifier:
    def __init__(self):
        self.province_clf = RandomForestClassifier(n_estimators=100)
        self.char_clf = RandomForestClassifier(n_estimators=100)
    
    def train(self, X_province, y_province, X_char, y_char):
        """
        训练省份和字符分类器
        """
        # 训练省份分类器
        X_train, X_test, y_train, y_test = train_test_split(
            X_province, y_province, test_size=0.2, random_state=42
        )
        self.province_clf.fit(X_train, y_train)
        province_acc = self.province_clf.score(X_test, y_test)
        
        # 训练字符分类器
        X_train, X_test, y_train, y_test = train_test_split(
            X_char, y_char, test_size=0.2, random_state=42
        )
        self.char_clf.fit(X_train, y_train)
        char_acc = self.char_clf.score(X_test, y_test)
        
        return province_acc, char_acc
    
    def predict(self, X_province, X_char):
        """
        预测车牌号码
        """
        province_pred = self.province_clf.predict(X_province.reshape(1, -1))
        char_preds = self.char_clf.predict(X_char)
        
        return province_pred[0], char_preds
    
    def save_models(self, province_path, char_path):
        """
        保存模型
        """
        joblib.dump(self.province_clf, province_path)
        joblib.dump(self.char_clf, char_path)
    
    def load_models(self, province_path, char_path):
        """
        加载模型
        """
        self.province_clf = joblib.load(province_path)
        self.char_clf = joblib.load(char_path) 