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

#### Using uv (recommended)

Install uv if you haven't already:

```bash
# Install uv
pip install uv
```

Then install dependencies for both the main project and the SCAPES submodule:

```bash
# Install main project dependencies
uv pip install -r requirements.txt

# Install SCAPES submodule dependencies
uv pip install -r modules/scapes/requirements.txt
```

#### Using pip

```bash
# Install main project dependencies
pip install -r requirements.txt

# Install SCAPES submodule dependencies
pip install -r modules/scapes/requirements.txt
```

### 3. Verify Installation

Run the main application to verify everything is installed correctly:

```bash
python app.py
```

Or use the inference interface:

```bash
python main.py
```
