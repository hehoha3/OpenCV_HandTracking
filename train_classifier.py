import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Tải dữ liệu đã được lưu từ file ./data.pickle và lưu vào biến data_dict
data_dict = pickle.load(open("./data.pickle", "rb"))

# Chuyển data và labels từ dictionary sang mảng NumPy.
data = np.asarray(data_dict["data"])
labels = np.asarray(data_dict["labels"])

# Chia dữ liệu thành tập train (x_train, y_train) và tập test (x_test, y_test) với tỷ lệ 80% - 20%.
# Dữ liệu được lộn xộn (shuffle=True) và giữ nguyên tỷ lệ phân phối của labels (stratify=labels)
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# train_test_split
model = RandomForestClassifier()

# Huấn luyện mô hình trên tập huấn luyện dựa vào phương thức fit
model.fit(x_train, y_train)

# Sử dụng mô hình đã huấn luyện để dự đoán label cho tập test
y_predict = model.predict(x_test)

# Tính độ chính xác của mô hình so với label thực tế
score = accuracy_score(y_predict, y_test)

# print ra tỷ lệ mẫu được phân loại đúng trên tập kiểm tra
print("{}% of samples were classified correctly !".format(score * 100))

# Mở file model.p để ghi, lưu mô hình đã huấn luyện dưới dạng pickle, và đóng file sau khi hoàn tất
f = open("model.p", "wb")
pickle.dump({"model": model}, f)
f.close()
