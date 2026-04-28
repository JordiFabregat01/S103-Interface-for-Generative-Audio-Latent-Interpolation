# S103-Interface-for-Generative-Audio-Latent-Interpolation

This is the repository for the group S103 during the course of Taller de Musicologia

## Quickstart

### 1. Clone the Repository with Submodules

This project includes the SCAPES submodule. Clone with submodules:

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/your-repo/S103-Interface-for-Generative-Audio-Latent-Interpolation.git

# Or if already cloned, initialize submodules
git submodule update --init --recursive
```

### 2. Install Dependencies

This project has two parts: a **Python backend** and a **TypeScript/Vite frontend**.

#### Backend (Python)

Install uv if you haven't already:

```bash
# Install uv
pip install uv
```

Then install dependencies for the backend and the SCAPES submodule:

```bash
# Install main project dependencies
uv pip install -r requirements.txt

# Install SCAPES submodule dependencies
uv pip install -r modules/scapes/requirements.txt
```

##### Using pip (alternative to uv)

```bash
# Install main project dependencies
pip install -r requirements.txt

# Install SCAPES submodule dependencies
pip install -r modules/scapes/requirements.txt
```

#### Frontend (Node.js)

The frontend is a Vite + TypeScript application. Install its dependencies:

```bash
# Install frontend dependencies
cd app/frontend
npm install
```

### 3. Verify Installation

#### Backend

Run the main application to verify everything is installed correctly:

```bash
# From the app/backend directory
cd app/backend
python -m uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

#### Frontend

To run the frontend development server:

```bash
# From the app/frontend directory
cd app/frontend
npm run dev
```

The frontend will be available at `http://localhost:5173` (or another port if that's in use).