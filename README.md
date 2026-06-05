# Skin Diseases Project

## Running the Web App

### Requirements

- Python 3.10+
- The trained model file `best_model_final.pt` placed inside `frontend`



### Setup

```bash
cd frontend

python -m venv .venv

pip install fastapi uvicorn torch torchvision pillow opencv-python numpy
```

### Run the frontend

```bash
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

### Run the other codes
they are just colabs so just executing them should be sufficient.
Taking phase1_checkpoint.pt from models directly should save you quite a lot of time.

uvicorn api:app --reload --port 8000
```

Open your browser at: **http://localhost:8000**

