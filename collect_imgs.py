import os

import cv2

# Tạo đường dẫn cơ sở cho thư mục chứa dữ liệu
DATA_DIR = "./data"
# Nếu không có thì tạo mới thư mực.
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Số lượng lớp dữ liệu cần thu thập (số folder -- mỗi folder chứa 1 cử chỉ tay)
number_of_classes = 3
# Số lượng hình ảnh cần thu thập cho mỗi folder
dataset_size = 100

# Mở camera mặc định (camera 0 -- camera máy tình tìm thấy được đầu tiên).
# TODO cap = cv2.VideoCapture('http://102.168.1.7:81/stream') ESP32-CAM
cap = cv2.VideoCapture(0)
#! Sẽ lặp lại 3 lần để thu đủ số folder của number_of_classes
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print("Collecting data for class {}".format(j))

    done = False
    while True:
        # đọc một khung hình từ camera
        # - ret là một biến boolean chỉ ra xem khung hình có được đọc thành công hay không
        # - frame chứa dữ liệu của khung hình
        ret, frame = cap.read()
        # Thêm văn bản lên khung hình (thông báo nhấn 'q' để chụp)
        cv2.putText(
            frame,
            'Ready? Press "Q" ! :)',
            (100, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )
        # Hàm imshow() hiển thị khung hình đã được chỉnh sửa trên một cửa sổ có tiêu đề "frame"
        cv2.imshow("frame", frame)
        # Kiểm tra phím nhấn và thoát vòng lặp
        if cv2.waitKey(25) == ord("q"):
            break

    # đếm số lượng hình ảnh đã thu thập
    counter = 0
    # Vòng lặp sẽ chạy đến khi số lượng hình ảnh đã thu thập đạt đến kích thước dữ liệu dataset_size.
    while counter < dataset_size:
        # Đọc frame và ret từ hàm read()
        ret, frame = cap.read()
        # Hiển thị khung hình đã được thu từ camera trên một cửa sổ có tiêu đề "frame"
        cv2.imshow("frame", frame)
        # * dừng chương trình lưu hình ảnh trong 25ms để đảm bảo rằng chương trình không bị đứng cứng do việc đọc khung hình liên tục
        cv2.waitKey(25)
        # lưu khung hình vào hệ thống tệp.
        cv2.imwrite(os.path.join(DATA_DIR, str(j), "{}.jpg".format(counter)), frame)

        counter += 1

cap.release()  # Đóng kết nối với camera
cv2.destroyAllWindows()  # Đóng tất cả các cửa sổ hiển thị đang mở
