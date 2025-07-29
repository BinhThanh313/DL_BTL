# Diabetic Retinopathy Detection System

## Mô tả dự án

Đây là hệ thống phát hiện và hỗ trợ chẩn đoán bệnh võng mạc tiểu đường từ ảnh đáy mắt, sử dụng deep learning với mô hình EfficientNet-B3 và kỹ thuật embedding. Dự án bao gồm các chức năng:
- Phân loại mức độ bệnh võng mạc tiểu đường (DR) từ ảnh đầu vào.
- Trực quan hóa vùng tổn thương bằng Grad-CAM.
- Trích xuất vector đặc trưng (embedding) cho ảnh.
- Tìm kiếm các ca bệnh tương đương dựa trên embedding.

## Cấu trúc thư mục
- `app.py`: Ứng dụng Streamlit giao diện người dùng.
- `train_classifier.py`: Script huấn luyện mô hình phân loại DR.
- `dr_best_model.pth`: Trọng số mô hình phân loại đã huấn luyện.
- `triplet_embedding.pth`: Trọng số mô hình embedding đã huấn luyện.
- `processed_images/`: Ảnh đã tiền xử lý.
- `aptos2019/`: Chứa dữ liệu gốc từ Kaggle (csv, ảnh train/test).
- `download_kaggle_aptos2019.ipynb`: Notebook hướng dẫn tải dữ liệu từ Kaggle.

## Hướng dẫn cài đặt
1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```
2. Tải dữ liệu từ Kaggle theo hướng dẫn trong notebook `download_kaggle_aptos2019.ipynb`.
3. Huấn luyện mô hình (nếu muốn):
   ```bash
   python train_classifier.py
   ```
4. Chạy ứng dụng:
   ```bash
   streamlit run app.py
   ```

## Sử dụng
- Upload ảnh đáy mắt lên giao diện web.
- Xem kết quả phân loại DR, vùng tổn thương (Grad-CAM), embedding và các ca bệnh tương đương.

## Dataset
- [Cuộc thi Kaggle: aptos2019-blindness-detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
- EfficientNet, Grad-CAM, Triplet Loss, Streamlit, PyTorch
