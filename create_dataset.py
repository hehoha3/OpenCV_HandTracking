import os
import pickle

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp

mp_hands = mp.solutions.hands  # Khởi tạo module xử lý bàn tay của MediaPipe
# cung cấp các tiện ích để vẽ các đường và hình dạng lên khung hình
mp_drawing = mp.solutions.drawing_utils
# Module cung cấp các styles vẽ mà ta có thể sử dụng để tùy chỉnh cách các đường được vẽ lên khung hình
mp_drawing_styles = mp.solutions.drawing_styles

# Khởi tạo đối tượng hands với chế độ static_image_mode ON và mức độ tin cậy phát hiện tối thiểu là 0.3.
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Định nghĩa đường dẫn cơ sở cho thư mục chứa dữ liệu
DATA_DIR = "./data"

# Danh sách để lưu trữ dữ liệu về các điểm đặc trưng của bàn tay
data = []
# Danh sách để lưu trữ label tương ứng với mỗi tập dữ liệu (label là tên thư mục chứa hình ảnh).
labels = []
# duyệt qua tất cả các thư mục con trong DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    # duyệt qua tất cả các hình ảnh trong mỗi thư mục con
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        # Danh sách tạm thời để lưu trữ dữ liệu về các điểm đặc trưng của bàn tay sau khi đã được xử lý.
        data_aux = []

        x_ = []  # Danh sách tạm thời để lưu trữ tọa độ x của các điểm đặc trưng
        y_ = []  # Danh sách tạm thời để lưu trữ tọa độ y của các điểm đặc trưng

        img = cv2.imread(
            os.path.join(DATA_DIR, dir_, img_path)
        )  # Đọc hình ảnh từ đường dẫn
        img_rgb = cv2.cvtColor(
            img, cv2.COLOR_BGR2RGB
        )  # Chuyển đổi màu sắc của hình ảnh từ BGR sang RGB để phù hợp với yêu cầu của MediaPipe

        # Xử lý hình ảnh đã chuyển đổi sang RGB bằng MediaPipe để phát hiện các điểm đặc trưng của bàn tay
        results = hands.process(img_rgb)
        # Kiểm tra xem có phát hiện được một hoặc nhiều bàn tay trên hình ảnh hay không
        if results.multi_hand_landmarks:
            # duyệt qua từng bàn tay mà MediaPipe phát hiện được trên hình ảnh
            for hand_landmarks in results.multi_hand_landmarks:
                # duyệt qua tất cả các điểm đặc trưng của một bàn tay
                for i in range(len(hand_landmarks.landmark)):
                    # Lấy tọa độ x và y của mỗi điểm đặc trưng
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    # thêm tọa độ x và y vào 2 list x_ và y_ tương ứng
                    x_.append(x)
                    y_.append(y)

                # duyệt qua tất cả các điểm đặc trưng của một bàn tay
                for i in range(len(hand_landmarks.landmark)):
                    # Lấy tọa độ x và y của mỗi điểm đặc trưng
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    # Đối với mỗi điểm đặc trưng, đoạn mã này tính toán sự khác biệt so với điểm đặc trưng có tọa độ x và y thấp nhất trên trục x và y tương ứng.
                    # Sự khác biệt này được tính bằng cách trừ đi giá trị nhỏ nhất của danh sách x_ và y_ tương ứng.
                    # Kết quả sau cùng được thêm vào danh sách data_aux
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Dữ liệu thu thập được (data_aux) được thêm vào danh sách data
            data.append(data_aux)
            # Nhãn tương ứng (thư mục chứa hình ảnh) được thêm vào danh sách labels
            labels.append(dir_)

# pickle.dump() lưu danh sách data và labels vào một file pickle tên là data.pickle
# Điều này cho phép lưu trữ dữ liệu dưới dạng binary, dễ dàng tái sử dụng và chia sẻ giữa các môi trường khác nhau
f = open("data.pickle", "wb")
pickle.dump({"data": data, "labels": labels}, f)
f.close()
