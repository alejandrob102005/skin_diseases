from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torchvision import transforms
from PIL import Image
import io
import base64
import numpy as np
import cv2

# =========================
# 1. CONFIGURACIÓN
# =========================
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
CLASS_LABELS = {
    'nv':    'Melanocytic Nevus',
    'mel':   'Melanoma',
    'bkl':   'Benign Keratosis',
    'bcc':   'Basal Cell Carcinoma',
    'akiec': 'Actinic Keratosis',
    'vasc':  'Vascular Lesion',
    'df':    'Dermatofibroma'
}
MALIGNANT = {'mel', 'bcc', 'akiec'}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

# =========================
# 2. MODELO
# =========================
def build_efficientnet_b2(num_classes: int = 7, freeze_backbone: bool = False) -> nn.Module:
    model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features  # 1408
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes)
    )
    return model

print(f"Cargando modelo en {DEVICE}...")

model = build_efficientnet_b2(num_classes=7, freeze_backbone=False)
model.load_state_dict(torch.load("models/efficientnet_b2_phase1.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("✅ Model ready")

# =========================
# 3. GRADCAM
# =========================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.gradients    = None
        self.activations  = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        x = x.to(DEVICE)
        output = self.model(x)
        self.model.zero_grad()
        output[0, class_idx].backward()

        grads       = self.gradients[0]                        # (C, H, W)
        activations = self.activations[0]                      # (C, H, W)
        weights     = grads.mean(dim=(1, 2))                   # (C,)

        cam = (weights[:, None, None] * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        return cam


gradcam_gen = GradCAM(model, model.features[-1])
print("✅ GradCAM ready")

# =========================
# 4. TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# =========================
# 5. HELPERS
# =========================
def preprocess(img_bytes: bytes):
    """Devuelve (img_np_uint8, tensor)."""
    img_pil    = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_np     = np.array(img_pil)                             # uint8 [0,255]
    tensor     = transform(img_pil).unsqueeze(0).to(DEVICE)
    return img_np, tensor

def run_prediction(tensor: torch.Tensor) -> dict:
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu()

    pred_idx   = probs.argmax().item()
    pred_code  = CLASS_NAMES[pred_idx]
    confidence = round(probs[pred_idx].item() * 100, 1)

    top3_idx = probs.topk(3).indices.tolist()
    top3 = [
        {
            "code":  CLASS_NAMES[i],
            "label": CLASS_LABELS[CLASS_NAMES[i]],
            "prob":  round(probs[i].item() * 100, 1)
        }
        for i in top3_idx
    ]

    return {
        "code":      pred_code,
        "label":     CLASS_LABELS[pred_code],
        "malignant": pred_code in MALIGNANT,
        "confidence": confidence,
        "top3":      top3
    }

def ndarray_to_base64(img_array: np.ndarray) -> str:
    pil_img = Image.fromarray(img_array.astype(np.uint8))
    buffer  = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# =========================
# 6. APP
# =========================
app = FastAPI(title="HAM10000 API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 7. ENDPOINTS
# =========================
@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}

@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Formato no soportado. Usa JPG, PNG o WEBP.")
    try:
        img_bytes          = await image.read()
        _, tensor          = preprocess(img_bytes)
        result             = run_prediction(tensor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {e}")
    return JSONResponse(content=result)

@app.post("/gradcam")
async def gradcam(image: UploadFile = File(...)):
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Formato no soportado. Usa JPG, PNG o WEBP.")
    try:
        img_bytes          = await image.read()
        img_np, tensor     = preprocess(img_bytes)
        prediction         = run_prediction(tensor)

        # GradCAM necesita gradientes → no usar torch.no_grad aquí
        pred_idx           = CLASS_NAMES.index(prediction["code"])
        cam_map            = gradcam_gen.generate(tensor, pred_idx)

        # Overlay
        img_float          = img_np.astype(np.float32) / 255.0
        heatmap            = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
        heatmap_rgb        = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        overlay            = np.uint8(255 * (0.5 * img_float + 0.5 * heatmap_rgb))

        original_b64       = ndarray_to_base64(img_np)
        gradcam_b64        = ndarray_to_base64(overlay)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar GradCAM: {e}")

    return JSONResponse(content={
        "prediction": prediction,
        "original":   original_b64,
        "gradcam":    gradcam_b64,
    })