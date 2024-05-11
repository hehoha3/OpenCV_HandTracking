import pickle

import cv2
import mediapipe as mp
import numpy as np

# Tải mô hình đã được huấn luyện từ file model.p và lưu vào biến model
model_dict = pickle.load(open("./model.p", "rb"))
model = model_dict["model"]

# Khởi động Camera
cap = cv2.VideoCapture(0)

# Khởi tạo đối tượng Hands từ MediaPipe để phát hiện tay trong video stream
mp_hands = mp.solutions.hands
# cung cấp các tiện ích để vẽ các đường và hình dạng lên khung hình
mp_drawing = mp.solutions.drawing_utils
# Module cung cấp các styles vẽ mà ta có thể sử dụng để tùy chỉnh cách các đường được vẽ lên khung hình
mp_drawing_styles = mp.solutions.drawing_styles

# Khởi tạo đối tượng hands với chế độ static_image_mode ON và mức độ tin cậy phát hiện tối thiểu là 0.3.
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# ánh xạ các số nguyên thành các ký tự -> chuyển đổi tín hiệu
labels_dict = {0: "A", 1: "B", 2: "L"}
while True:

    # Khởi tạo ba danh sách trống, giống với create_dataset
    data_aux = []
    x_ = []
    y_ = []

    # Đọc một khung hình từ camera và lưu vào biến frame. ret
    ret, frame = cap.read()

    # Lấy chiều cao (H) và chiều rộng (W) của khung hình
    H, W, _ = frame.shape
    # Chuyển đổi màu sắc của khung hình từ BGR -> RGB (được sử dụng bởi MediaPipe).
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Xử lý khung hình đã chuyển đổi màu sắc với MediaPipe Hands để phát hiện tay và vẽ các điểm đặc trưng
    results = hands.process(frame_rgb)
    # Kiểm tra xem có phát hiện được tay hay không
    if results.multi_hand_landmarks:
        # Duyệt qua từng tay được phát hiện
        for hand_landmarks in results.multi_hand_landmarks:
            # Vẽ các điểm đặc trưng và kết nối giữa các điểm trên tay lên khung hình
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

        # Duyệt qua từng tay được phát hiện
        for hand_landmarks in results.multi_hand_landmarks:
            # Lấy tọa độ x và y của từng điểm đặc trưng trên tay và lưu vào danh sách x_ và y_
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            # Tính toán sự dịch chuyển của tọa độ x và y so với tọa độ nhỏ nhất trong danh sách x_ và y_, và thêm vào data_aux
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Tính toán tọa độ góc trái trên cùng của khung bao quanh tay
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        # Tính toán tọa độ góc phải dưới cùng của khung bao quanh tay
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Sử dụng model đã huấn luyện để dự đoán ký tự dựa trên dữ liệu đặc trưng thu được từ tay.
        prediction = model.predict([np.asarray(data_aux)])

        # Chuyển đổi số nguyên dự đoán từ mô hình thành ký tự tương ứng thông qua dictionary labels_dict
        predicted_character = labels_dict[int(prediction[0])]

        # Vẽ một khung hình xung quanh tay trên khung hình
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        # Vẽ ký tự dự đoán lên khung hình.
        cv2.putText(
            frame,
            predicted_character,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )

    # Hiển thị khung hình đã xử lý và chờ 1 millisecond trước khi đọc khung hình tiếp theo.

    cv2.imshow("frame", frame)
    cv2.waitKey(1)

# Khi vòng lặp kết thúc, đoạn code này sẽ đóng camera và hủy tất cả các cửa sổ hiển thị
cap.release()
cv2.destroyAllWindows()
