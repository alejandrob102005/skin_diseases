# Skin Diseases Project

## Running the Web App

### Requirements

- Python 3.10+
- The trained model file `best_model_final.pt`

### Setup

```bash
cd frontend
python -m venv .venv
pip install fastapi uvicorn torch torchvision pillow opencv-python numpy
```

### Activate the virtual environment

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### Run the server

```bash
uvicorn api:app --reload --port 8000
```

Open your browser at: **http://localhost:8000**

---

## Running the Other Notebooks

The remaining code is structured as Google Colab notebooks — simply open and execute them sequentially.

> **Tip:** Using `phase1_checkpoint.pt` from the `models/` directory directly will save significant training time.
