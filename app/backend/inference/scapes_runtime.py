from __future__ import annotations

import importlib.util
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCAPES_ROOT = REPO_ROOT / "modules" / "scapes"

if str(SCAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SCAPES_ROOT))


def _load_module(module_name: str, file_path: Path):
    logger.debug(f"Loading external module '{module_name}' from {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        logger.error(f"Failed to load module '{module_name}' from {file_path}")
        raise ImportError(f"Unable to load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


logger.info(f"Initializing SCAPES runtime from {SCAPES_ROOT}")
_clap_module = _load_module("_backend_scapes_clap_wrapper", SCAPES_ROOT / "SCAPES" / "auxiliar" / "clap_wrapper.py")
_encodec_module = _load_module("_backend_scapes_encodec_wrapper", SCAPES_ROOT / "SCAPES" / "auxiliar" / "encodec_wrapper.py")
_flow_inference_module = _load_module("_backend_scapes_flow_inference", SCAPES_ROOT / "SCAPES" / "inference" / "FlowInference.py")
_local_encoder_module = _load_module("_backend_scapes_local_encoder", SCAPES_ROOT / "SCAPES" / "models" / "factorization" / "LocalEncoder.py")
_flow_module = _load_module("_backend_scapes_flow_model", SCAPES_ROOT / "SCAPES" / "models" / "flow" / "FlowModel.py")

CLAPWrapper = _clap_module.CLAPWrapper
EncodecProcessor = _encodec_module.EncodecProcessor
FlowInference = _flow_inference_module.FlowInference
run_interpolation_pipeline = _flow_inference_module.run_interpolation_pipeline
load_local_encoder = _local_encoder_module.load_local_encoder
load_flow_model = _flow_module.load_flow_model