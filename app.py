import streamlit as st

from ultralytics import YOLO

import torch

import torch.nn as nn

from torchvision import models, transforms

from PIL import Image, ImageDraw, ImageFont

import torch.nn.functional as F

import cv2

import tempfile

import numpy as np



# ==========================================

# 1. THIẾT LẬP GIAO DIỆN TRANG WEB

# ==========================================

st.set_page_config(page_title="Nhận Diện Biển Báo", page_icon="🚦", layout="wide")

st.title("🚦 HỆ THỐNG NHẬN DIỆN BIỂN BÁO GIAO THÔNG")

st.markdown("**Đồ án môn học** - Sử dụng kết hợp YOLOv11 và Mô hình Phân loại (ResNet18+CBAM / ViT)")



# ==========================================

# 2. ĐỊNH NGHĨA KIẾN TRÚC MẠNG & CLASS_NAMES

# ==========================================

# 2.1. Kiến trúc ResNet18 + CBAM

class CBAM(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16):

        super(CBAM, self).__init__()

        self.channel_att = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(gate_channels, gate_channels // reduction_ratio, 1, bias=False),

            nn.ReLU(),

            nn.Conv2d(gate_channels // reduction_ratio, gate_channels, 1, bias=False),

            nn.Sigmoid()

        )

        self.spatial_att = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False), nn.Sigmoid())

    def forward(self, x):

        x = x * self.channel_att(x)

        avg_out = torch.mean(x, dim=1, keepdim=True)

        max_out, _ = torch.max(x, dim=1, keepdim=True)

        spatial = torch.cat([avg_out, max_out], dim=1)

        x = x * self.spatial_att(spatial)

        return x



def get_resnet_model(num_classes):

    model = models.resnet18()

    model.layer4.add_module("cbam_block", CBAM(512))

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model



# 2.2. Kiến trúc Vision Transformer (ViT)

def get_vit_model(num_classes):

    # Sử dụng ViT-B/16 mặc định từ torchvision

    model = models.vit_b_16()

    # Thay thế lớp classification head cuối cùng cho phù hợp số class

    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    return model



CLASS_NAMES = [

    'Đường người đi bộ cắt ngang',

    'Đường giao nhau (ngã ba bên phải)',

    'Cấm đi ngược chiều',

    'Phải đi vòng sang bên phải',

    'Giao nhau với đường đồng cấp',

    'Giao nhau với đường không ưu tiên',

    'Chỗ ngoặt nguy hiểm vòng bên trái',

    'Cấm rẽ trái',

    'Bến xe buýt',

    'Nơi giao nhau chạy theo vòng xuyến',

    'Cấm dừng và đỗ xe',

    'Chỗ quay xe',

    'Biển gộp làn đường theo phương tiện',

    'Đi chậm',

    'Cấm xe tải',

    'Đường bị thu hẹp về phía phải',

    'Giới hạn chiều cao',

    'Cấm quay đầu',

    'Cấm ô tô khách và ô tô tải',

    'Cấm rẽ phải và quay đầu',

    'Cấm ô tô',

    'Đường bị thu hẹp về phía trái',

    'Gồ giảm tốc phía trước',

    'Cấm xe hai và ba bánh',

    'Kiểm tra',

    'Chỉ dành cho xe máy*',

    'Chướng ngoại vật phía trước',

    'Trẻ em',

    'Xe tải và xe công*',

    'Cấm mô tô và xe máy',

    'Chỉ dành cho xe tải*',

    'Đường có camera giám sát',

    'Cấm rẽ phải',

    'Nhiều chỗ ngoặt nguy hiểm liên tiếp, chỗ đầu tiên sang phải',

    'Cấm xe sơ-mi rơ-moóc',

    'Cấm rẽ trái và phải',

    'Cấm đi thẳng và rẽ phải',

    'Đường giao nhau (ngã ba bên trái)',

    'Giới hạn tốc độ (50km/h)',

    'Giới hạn tốc độ (60km/h)',

    'Giới hạn tốc độ (80km/h)',

    'Giới hạn tốc độ (40km/h)',

    'Các xe chỉ được rẽ trái',

    'Chiều cao tĩnh không thực tế',

    'Nguy hiểm khác',

    'Đường một chiều',

    'Cấm đỗ xe',

    'Cấm ô tô quay đầu xe (được rẽ trái)',

    'Giao nhau với đường sắt có rào chắn',

    'Cấm rẽ trái và quay đầu xe',

    'Chỗ ngoặt nguy hiểm vòng bên phải',

    'Chú ý chướng ngại vật – vòng tránh sang bên phải'

]

# ==========================================

# 3. HÀM TẢI MÔ HÌNH

# ==========================================

@st.cache_resource

def load_models():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   

    # Load YOLO

    yolo_model = YOLO("models/best.pt")

   

    # Load ResNet18 + CBAM

    resnet_model = get_resnet_model(len(CLASS_NAMES)).to(device)

    resnet_model.load_state_dict(torch.load("models/resnet_cbam_final.pth", map_location=device, weights_only=True))

    resnet_model.eval()

   

    # Load ViT

    vit_model = get_vit_model(len(CLASS_NAMES)).to(device)

    vit_model.load_state_dict(torch.load("models/vit_final.pth", map_location=device, weights_only=True))

    vit_model.eval()

   

    return yolo_model, resnet_model, vit_model, device



yolo, resnet, vit, device = load_models()



transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])



def draw_text_pil(img, text, position, font_path="C:/Windows/Fonts/arial.ttf", font_size=18, color=(255, 0, 0), stroke_width=1, stroke_fill=(0,0,0)):

    try:

        font = ImageFont.truetype(font_path, font_size)

    except Exception:

        font = ImageFont.load_default()



    pil_img = Image.fromarray(img)

    draw = ImageDraw.Draw(pil_img)

    try:

        draw.text(position, text, font=font, fill=tuple(color), stroke_width=stroke_width, stroke_fill=tuple(stroke_fill))

    except TypeError:

        draw.text(position, text, font=font, fill=tuple(color))



    return np.array(pil_img)



# ==========================================

# 4. THANH ĐIỀU HƯỚNG VÀ XỬ LÝ

# ==========================================

app_mode = st.sidebar.radio("CHỌN CHẾ ĐỘ TEST:", ["🖼️ Nhận diện Ảnh tĩnh", "🎥 Nhận diện Video"])

st.sidebar.markdown("---")



# Thêm lựa chọn mô hình phân loại (Classification Model)

classifier_choice = st.sidebar.radio("🤖 CHỌN MÔ HÌNH PHÂN LOẠI:", ["YOLO + ResNet18 (CBAM)", "YOLO + Vision Transformer (ViT)"])

st.sidebar.markdown("---")



# Xác định mô hình đang được sử dụng

active_classifier = resnet if classifier_choice == "YOLO + ResNet18 (CBAM)" else vit



# ------------------------------------------

# CHẾ ĐỘ 1: XỬ LÝ ẢNH TĨNH

# ------------------------------------------

if app_mode == "🖼️ Nhận diện Ảnh tĩnh":

    uploaded_file = st.sidebar.file_uploader("📤 Tải ảnh giao thông lên", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:

            st.subheader("📸 Ảnh gốc")

            st.image(image, use_container_width=True)

       

        with st.spinner(f"⏳ Đang xử lý bằng {classifier_choice}..."):

            results = yolo.predict(image, conf=0.25, device=0 if torch.cuda.is_available() else 'cpu')

            boxes = results[0].boxes.xyxy.cpu().numpy()

           

            with col2:

                st.subheader("🎯 Kết quả nhận diện")

                if len(boxes) == 0:

                    st.warning("Không tìm thấy biển báo!")

                else:

                    for i, box in enumerate(boxes):

                        x1, y1, x2, y2 = map(int, box)

                        cropped_img = image.crop((x1, y1, x2, y2))

                       

                        img_tensor = transform(cropped_img).unsqueeze(0).to(device)

                        with torch.no_grad():

                            # Sử dụng mô hình đã chọn (ResNet hoặc ViT)

                            outputs = active_classifier(img_tensor)

                            probs = F.softmax(outputs, dim=1)

                            conf, predicted = torch.max(probs, 1)

                       

                        label = CLASS_NAMES[predicted.item()]

                        st.markdown("---")

                        sub_col1, sub_col2 = st.columns([1, 3])

                        with sub_col1:

                            st.image(cropped_img, width=100)

                        with sub_col2:

                            st.success(f"**{label}** ({conf.item()*100:.1f}%)")



# ------------------------------------------

# CHẾ ĐỘ 2: XỬ LÝ VIDEO TRỰC TIẾP

# ------------------------------------------

elif app_mode == "🎥 Nhận diện Video":

    uploaded_video = st.sidebar.file_uploader("📤 Tải video (.mp4, .avi) lên", type=['mp4', 'avi', 'mov'])

    record_conf_threshold = st.sidebar.slider("Ngưỡng lưu biển báo (%)", 0, 100, 60)

   

    if uploaded_video is not None:

        tfile = tempfile.NamedTemporaryFile(delete=False)

        tfile.write(uploaded_video.read())

       

        cap = cv2.VideoCapture(tfile.name)

       

        st.subheader(f"🎥 Trình phát Video - Đang dùng: {classifier_choice}")

        stframe = st.empty()

        detections_summary = {}

        stop_button = st.button("🛑 Dừng phát video")

       

        while cap.isOpened() and not stop_button:

            ret, frame = cap.read()

            if not ret:

                st.success("✅ Đã phát hết video!")

                break

               

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = yolo.predict(frame_rgb, conf=0.25, device=0 if torch.cuda.is_available() else 'cpu', verbose=False)

            boxes = results[0].boxes.xyxy.cpu().numpy()

           

            for box in boxes:

                x1, y1, x2, y2 = map(int, box)

                crop = frame_rgb[y1:y2, x1:x2]

               

                if crop.size != 0:

                    pil_img = Image.fromarray(crop)

                    img_tensor = transform(pil_img).unsqueeze(0).to(device)

                   

                    with torch.no_grad():

                        # Sử dụng mô hình đã chọn

                        outputs = active_classifier(img_tensor)

                        conf, predicted = torch.max(F.softmax(outputs, dim=1), 1)

                       

                    label = CLASS_NAMES[predicted.item()]

                    score = conf.item() * 100

                   

                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    text = f"{label} {score:.1f}%"

                    frame_rgb = draw_text_pil(frame_rgb, text, (x1, max(y1 - 10, 10)), font_size=16, color=(255, 0, 0), stroke_width=2, stroke_fill=(0,0,0))



                    if score >= record_conf_threshold:

                        try:

                            sample_pil = pil_img.copy()

                        except Exception:

                            sample_pil = Image.fromarray(crop)

                        sample_pil = sample_pil.resize((200, 200))



                        info = detections_summary.get(label)

                        if info is None:

                            detections_summary[label] = {"best_score": score, "count": 1, "sample": sample_pil}

                        else:

                            info["count"] += 1

                            if score > info["best_score"]:

                                info["best_score"] = score

                                info["sample"] = sample_pil

           

            stframe.image(frame_rgb, channels="RGB", use_container_width=True)

           

        cap.release()



        st.markdown("---")

        st.subheader("📋 Tổng hợp biển báo phát hiện (sau khi video kết thúc)")

        if len(detections_summary) == 0:

            st.info(f"Không có biển báo nào đạt ngưỡng {record_conf_threshold}%.")

        else:

            items = sorted(detections_summary.items(), key=lambda x: x[1]["best_score"], reverse=True)

            for label, info in items:

                cols = st.columns([1, 3])

                with cols[0]:

                    st.image(info["sample"], use_container_width=True)

                with cols[1]:

                    st.markdown(f"**{label}**")

                    st.write(f"Best: {info['best_score']:.1f}% — Count: {info['count']}") 

