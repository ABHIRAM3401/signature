import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# Configuration
IMAGE_SIZE = (155, 220)
EMBEDDING_DIM = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=512):
        super(SiameseNetwork, self).__init__()

        self.backbone = models.mobilenet_v2(weights=None)

        self.backbone.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.backbone.classifier = nn.Identity()

        self.embedding = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, embedding_dim),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward_one(self, x):
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, 1)
            features = features.view(features.size(0), -1)
        return self.embedding(features)

    def forward(self, img1, img2):
        emb1 = self.forward_one(img1)
        emb2 = self.forward_one(img2)
        diff = torch.abs(emb1 - emb2)
        return self.classifier(diff).squeeze()


# Load model
print("Loading model...")
model = SiameseNetwork(EMBEDDING_DIM)
model_path = "best_model.pth"

if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded!")
else:
    print("⚠️ WARNING: model not found, random weights used")

model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def verify_signatures(original_image, test_image):
    if original_image is None or test_image is None:
        return "❌ Please upload both images!"

    try:
        img1 = original_image.convert('L')
        img2 = test_image.convert('L')

        img1_tensor = transform(img1).unsqueeze(0).to(DEVICE)
        img2_tensor = transform(img2).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            score = model(img1_tensor, img2_tensor).item()

        is_match = score > 0.5

        if score > 0.8 or score < 0.2:
            confidence = "High"
        elif 0.6 < score < 0.8 or 0.2 < score < 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"

        if is_match:
            result = f"✅ MATCH!\n\nSame person\n\n"
        else:
            result = f"❌ NO MATCH!\n\nDifferent persons\n\n"

        result += f"Similarity Score: {score:.2%}\n"
        result += f"Confidence: {confidence}"
        return result

    except Exception as e:
        return f"❌ Error: {str(e)}"


demo = gr.Interface(
    fn=verify_signatures,
    inputs=[
        gr.Image(type="pil", label="Original Signature"),
        gr.Image(type="pil", label="Test Signature")
    ],
    outputs=gr.Textbox(label="Result", lines=6),
    title="🛡️ SignatureGuard"
)


if __name__ == "__main__":
    demo.launch()
