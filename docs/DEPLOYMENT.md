# TruthLens — Deployment & Operations Guide

This guide covers running TruthLens via Docker Compose, running locally with native Python environments, and production deployment considerations.

---

## 🐳 Option A: Single-Command Docker Compose (Recommended)

Docker Compose spins up the MongoDB database, FastAPI backend, and Streamlit frontend in isolated containers on a shared bridge network.

### Prerequisites
- Docker Engine $\ge 20.10$
- Docker Compose v2

### Commands
```bash
# 1. Clone the repository and enter directory
cd truthLensCapstone

# 2. Build and start all three services in detached mode
docker compose up --build -d

# 3. View streaming logs across all containers
docker compose logs -f

# 4. Tear down containers and volumes
docker compose down -v
```

### Container Endpoints
| Service | URL | Description |
| :--- | :--- | :--- |
| **Frontend** | `http://localhost:8501` | Streamlit User Interface |
| **Backend API** | `http://localhost:8000` | FastAPI REST Gateway |
| **Swagger Docs** | `http://localhost:8000/docs` | Interactive OpenAPI documentation |
| **MongoDB** | `localhost:27017` | Document Database |

---

## 💻 Option B: Running Locally (Native Python)

If developing without Docker or on machines without Docker installed:

### 1. Setup Virtual Environment
```bash
# Create and activate virtual environment
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
# Install backend, frontend, and pipeline requirements
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
pip install -r cv-pipeline/requirements.txt
pip install -r nlp-pipeline/requirements.txt
```

### 3. Launch the Backend
```bash
# In Terminal 1:
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
> **Note on MongoDB:** If MongoDB is running on `localhost:27017`, the backend connects automatically. If MongoDB is offline, TruthLens automatically activates its **in-memory async storage fallback**, allowing full end-to-end functionality without errors!

### 4. Launch the Streamlit Frontend
```bash
# In Terminal 2:
streamlit run frontend/app.py --server.port 8501
```

Access the UI at `http://localhost:8501`.

---

## ⚙️ Environment Variables Reference

### Backend (`/backend/.env`)
| Variable | Default | Description |
| :--- | :--- | :--- |
| `MONGODB_URL` | `mongodb://localhost:27017` | MongoDB connection URI (`mongodb://mongo:27017` in Docker) |
| `DATABASE_NAME` | `truthlens_db` | MongoDB database name |
| `UPLOAD_DIR` | `./uploads` | Directory for temporarily storing uploaded media files |
| `STATIC_DIR` | `./static` | Root directory for static file serving |
| `HEATMAP_DIR` | `./static/heatmaps` | Output directory for generated Grad-CAM heatmaps |
| `HOST` | `0.0.0.0` | Binding host address |
| `PORT` | `8000` | Binding network port |

### Frontend (`/frontend/.env`)
| Variable | Default | Description |
| :--- | :--- | :--- |
| `BACKEND_URL` | `http://localhost:8000` | Target URL of the FastAPI backend (`http://backend:8000` in Docker) |
