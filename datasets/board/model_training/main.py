from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).parent

# Load pretrained model (auto-downloaded on first run)
model = YOLO('yolo11s.pt')

# Train
results = model.train(
    data=str(HERE / 'data.yaml'),
    epochs=100,
    imgsz=640,
    device='cpu',
    batch=8,
    patience=20,
    name='pirairuchess_v1'
)
