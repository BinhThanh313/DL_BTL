
# --- STREAMLIT DIABETIC RETINOPATHY SYSTEM ---
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from torchvision.models import efficientnet_b3

# 1. Load DR Classifier (EfficientNet-B3, num_classes=5)
@st.cache_resource
def load_classifier():
    model = efficientnet_b3(weights=None, num_classes=5)
    state_dict = torch.load('dr_best_model.pth', map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

# 2. Load Embedding Model (EfficientNet-B3 backbone, Linear(1536,128))
class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.backbone = efficientnet_b3(weights=None)
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, embedding_dim)
    def forward(self, x):
        return self.backbone(x)

@st.cache_resource
def load_embedding_model():
    model = EmbeddingNet(embedding_dim=128)
    state_dict = torch.load('triplet_embedding.pth', map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

# 3. Preprocess image for model input
def preprocess_image(img):
    img = img.resize((300, 300))
    img_np = np.array(img) / 255.0
    if img_np.ndim == 2:
        img_np = np.stack([img_np]*3, axis=-1)
    elif img_np.shape[2] == 4:
        img_np = img_np[..., :3]
    img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return img_tensor

# 4. Predict DR class
def predict_class(img, model):
    img_tensor = preprocess_image(img)
    with torch.no_grad():
        output = model(img_tensor)
        dr_level = int(torch.argmax(output, dim=1).item())
    return dr_level

# 5. Grad-CAM visualization (using pytorch-grad-cam)
def get_gradcam(img, model):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    img_tensor = preprocess_image(img)
    img_tensor.requires_grad_()
    # Use last feature layer of EfficientNet-B3
    target_layer = model.features[-1] if hasattr(model, 'features') else model.backbone.features[-1]
    # Predict class
    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()
    # Grad-CAM
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=img_tensor, targets=[ClassifierOutputTarget(pred_class)])
    grayscale_cam = grayscale_cam[0]
    rgb_img = np.array(img.resize((300, 300))) / 255.0
    if rgb_img.ndim == 2:
        rgb_img = np.stack([rgb_img]*3, axis=-1)
    elif rgb_img.shape[2] == 4:
        rgb_img = rgb_img[..., :3]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    cam_image = Image.fromarray(cam_image).resize(img.size)
    return cam_image

# 6. Extract embedding vector
def extract_embedding(img, model):
    img_tensor = preprocess_image(img)
    with torch.no_grad():
        embedding = model(img_tensor)
    embedding_vec = embedding.squeeze().cpu().numpy()
    return embedding_vec


# --- STREAMLIT UI ---

st.title("Hệ thống chẩn đoán võng mạc tiểu đường từ ảnh đáy mắt")
uploaded_file = st.file_uploader("Chọn ảnh đáy mắt", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Ảnh đầu vào", use_container_width=True)

    classifier = load_classifier()
    embedding_model = load_embedding_model()

    # 1. DR Classification
    dr_level = predict_class(img, classifier)
    st.write(f"Kết quả phân loại DR: {dr_level}")

    # 2. Grad-CAM segmentation
    gradcam_img = get_gradcam(img, classifier)
    st.image(gradcam_img, caption="Grad-CAM phân đoạn tổn thương", use_container_width=True)

    # 3. Embedding extraction
    embedding_vec = extract_embedding(img, embedding_model)

    # 4. Hiển thị các ảnh bệnh tương đương (gần nhất về embedding, cùng mức độ DR)
    st.header("Các ảnh bệnh tương đương (gần nhất về embedding, cùng mức độ DR)")
    import os
    import pandas as pd
    img_dir = "processed_images"
    train_csv = "aptos2019/train.csv"
    if not os.path.exists(train_csv):
        st.warning("Không tìm thấy file aptos2019/train.csv. Vui lòng kiểm tra lại dữ liệu!")
    elif not os.path.exists(img_dir):
        st.warning("Không tìm thấy thư mục processed_images. Vui lòng kiểm tra lại dữ liệu!")
    else:
        try:
            df = pd.read_csv(train_csv)
            # processed_images chỉ là ảnh đã tiền xử lý, dataset gốc là aptos2019/train.csv
            # Lấy danh sách id_code có file ảnh đã xử lý
            processed_files = {f[:-4] for f in os.listdir(img_dir) if f.endswith('.png')}
            # Lọc các ảnh có cùng mức độ DR và đã được xử lý
            same_dr_imgs = [f for f in df[df['diagnosis'] == dr_level]['id_code'].tolist() if f in processed_files]
            if not same_dr_imgs:
                st.info("Không có ảnh nào cùng mức độ DR đã được xử lý.")
                st.write(f"Số lượng id_code có file ảnh đã xử lý: {len(processed_files)}")
                st.write(f"Số lượng ảnh cùng mức độ DR trong train.csv: {len(df[df['diagnosis'] == dr_level])}")
                st.write(f"Danh sách id_code cùng mức độ DR: {df[df['diagnosis'] == dr_level]['id_code'].tolist()[:10]}")
                st.write(f"Danh sách id_code đã xử lý: {list(processed_files)[:10]}")
            else:
                # Giới hạn số lượng ảnh duyệt để tăng tốc
                MAX_COMPARE = 200
                if len(same_dr_imgs) > MAX_COMPARE:
                    st.info(f"Chỉ so sánh {MAX_COMPARE} ảnh gần nhất trong {len(same_dr_imgs)} ảnh cùng mức độ DR.")
                    same_dr_imgs = same_dr_imgs[:MAX_COMPARE]
                N = min(5, len(same_dr_imgs))
                dists = []
                # Batch embedding để tăng tốc
                batch_imgs = []
                batch_names = []
                for fname in same_dr_imgs:
                    img_path = os.path.join(img_dir, fname + '.png')
                    try:
                        batch_imgs.append(Image.open(img_path))
                        batch_names.append(fname)
                    except Exception:
                        continue
                # Tính embedding cho batch
                batch_tensors = [preprocess_image(im) for im in batch_imgs]
                if batch_tensors:
                    batch_tensor = torch.cat(batch_tensors, dim=0)
                    with torch.no_grad():
                        batch_emb = embedding_model(batch_tensor).cpu().numpy()
                    for i, fname in enumerate(batch_names):
                        dist = np.linalg.norm(embedding_vec - batch_emb[i])
                        dists.append((dist, fname))
                if not dists:
                    st.info("Không thể tính embedding cho các ảnh tương đương.")
                else:
                    dists.sort()
                    top_imgs = [fname for _, fname in dists[:N]]
                    imgs = []
                    captions = []
                    for fname in top_imgs:
                        img_path = os.path.join(img_dir, fname + '.png')
                        try:
                            imgs.append(Image.open(img_path))
                            # Lấy thông tin bệnh án từ df
                            info = df[df['id_code'] == fname].iloc[0]
                            captions.append(f"ID: {fname}\nDR: {info['diagnosis']}")
                        except Exception:
                            imgs.append(None)
                            captions.append(f"ID: {fname}")
                    st.write(f"{N} bệnh án tương đương (cùng mức độ DR):")
                    st.image([im for im in imgs if im is not None], caption=captions, width=120, use_container_width=True)
        except Exception as e:
            st.error(f"Lỗi khi đọc dữ liệu: {e}")