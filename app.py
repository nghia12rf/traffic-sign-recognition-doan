import streamlit as st
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

# ==========================================
# 1. THIẾT LẬP GIAO DIỆN TRANG WEB
# ==========================================
st.set_page_config(page_title="Nhận Diện Biển Báo", page_icon="🚦", layout="wide")
st.title("🚦 HỆ THỐNG NHẬN DIỆN BIỂN BÁO GIAO THÔNG")
st.markdown("**Đồ án môn học** - Sử dụng kết hợp YOLOv11 và ResNet18 + CBAM Attention")

# ==========================================
# 2. ĐỊNH NGHĨA KIẾN TRÚC MẠNG RESNET + CBAM
# ==========================================
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

def get_model(num_classes):
    model = models.resnet18()
    model.layer4.add_module("cbam_block", CBAM(512))
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

CLASS_NAMES = ['Biển gộp làn đường theo phương tiện', 'Bến xe buýt', 'Chiều cao tĩnh không thực tế', 'Chú ý chướng ngại vật – vòng tránh sang bên phải', 'Chướng ngoại vật phía trước', 'Chỉ dành cho xe máy*', 'Chỉ dành cho xe tải*', 'Chỗ ngoặt nguy hiểm vòng bên phải', 'Chỗ ngoặt nguy hiểm vòng bên trái', 'Chỗ quay xe', 'Các xe chỉ được rẽ trái', 'Cấm dừng và đỗ xe', 'Cấm mô tô và xe máy', 'Cấm quay đầu', 'Cấm rẽ phải', 'Cấm rẽ phải và quay đầu', 'Cấm rẽ trái', 'Cấm rẽ trái và phải', 'Cấm rẽ trái và quay đầu xe', 'Cấm xe hai và ba bánh', 'Cấm xe sơ-mi rơ-moóc', 'Cấm xe tải', 'Cấm ô tô', 'Cấm ô tô khách và ô tô tải', 'Cấm ô tô quay đầu xe (được rẽ trái)', 'Cấm đi ngược chiều', 'Cấm đi thẳng và rẽ phải', 'Cấm đỗ xe', 'Giao nhau với đường không ưu tiên', 'Giao nhau với đường sắt có rào chắn', 'Giao nhau với đường đồng cấp', 'Giới hạn chiều cao', 'Gồ giảm tốc phía trước', 'Kiểm tra', 'Nguy hiểm khác', 'Nhiều chỗ ngoặt nguy hiểm liên tiếp, chỗ đầu tiên sang phải', 'Nơi giao nhau chạy theo vòng xuyến', 'Phải đi vòng sang bên phải', 'Trẻ em', 'Xe tải và xe công*', 'Đi chậm', 'Đường bị thu hẹp về phía phải', 'Đường bị thu hẹp về phía trái', 'Đường có camera giám sát', 'Đường giao nhau (ngã ba bên phải)', 'Đường giao nhau (ngã ba bên trái)', 'Đường một chiều', 'Đường người đi bộ cắt ngang']

# ==========================================
# 3. HÀM TẢI MÔ HÌNH (CACHE LÊN RAM/GPU)
# ==========================================
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load YOLO
    yolo_model = YOLO("models/best.pt")
    
    # Load ResNet
    resnet_model = get_model(len(CLASS_NAMES)).to(device)
    resnet_model.load_state_dict(torch.load("models/resnet_cbam_final.pth", map_location=device, weights_only=True))
    resnet_model.eval()
    
    return yolo_model, resnet_model, device

yolo, resnet, device = load_models()

# Tiền xử lý ảnh cho ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 4. GIAO DIỆN UPLOAD VÀ XỬ LÝ
# ==========================================
uploaded_file = st.sidebar.file_uploader("📤 Tải ảnh giao thông lên đây", type=['jpg', 'jpeg', 'png'])
st.sidebar.markdown("---")
st.sidebar.info("Hệ thống sẽ tự động quét bằng GPU và phân loại biển báo.")

if uploaded_file is not None:
    # Đọc ảnh gốc
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📸 Ảnh gốc")
        st.image(image, use_container_width=True)
    
    with st.spinner("⏳ Đang quét biển báo (YOLO) và phân loại (ResNet)..."):
        # Chạy YOLO
        results = yolo.predict(image, conf=0.25, device=0 if torch.cuda.is_available() else 'cpu')
        boxes = results[0].boxes.xyxy.cpu().numpy() # Lấy tọa độ [x1, y1, x2, y2]
        
        with col2:
            st.subheader("🎯 Kết quả nhận diện")
            
            if len(boxes) == 0:
                st.warning("Không tìm thấy biển báo nào trong ảnh!")
            else:
                # Xử lý từng biển báo được cắt ra
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Cắt vùng ảnh chứa biển báo
                    cropped_img = image.crop((x1, y1, x2, y2))
                    
                    # Đưa vào ResNet dự đoán
                    img_tensor = transform(cropped_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = resnet(img_tensor)
                        probs = F.softmax(outputs, dim=1)
                        conf, predicted = torch.max(probs, 1)
                    
                    label = CLASS_NAMES[predicted.item()]
                    score = conf.item() * 100
                    
                    # Hiển thị trực quan từng biển báo
                    st.markdown("---")
                    sub_col1, sub_col2 = st.columns([1, 3])
                    with sub_col1:
                        st.image(cropped_img, caption=f"Biển báo #{i+1}", width=120)
                    with sub_col2:
                        st.success(f"**Tên biển báo:** {label}")
                        st.info(f"**Độ tin cậy (Confidence):** {score:.2f}%")