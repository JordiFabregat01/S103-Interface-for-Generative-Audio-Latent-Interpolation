import torch
from pathlib import Path

from inference.methods import get_inference_engine
from inference.constants import _get_audio_asset_path
from inference.models import AudioElement


def precompute():
    engine = get_inference_engine()
    cwd = Path.cwd().resolve()

    for audio in AudioElement:
        audio_path = _get_audio_asset_path(audio).resolve()
        stem = audio.value

        print(f"\nProcessing: {audio}")

        # 1. Load audio
        audio_tensor = engine.load_audio_to_tensor(str(audio_path))

        # 2. Encode into atoms
        atoms = engine.encode_audio_to_atoms(audio_tensor)

        # 3. Compute contexts
        contexts = engine.compute_context_track(atoms)

        # 4. Set output cache paths
        # Keep canonical cache in assets, and mirror in cwd because FlowInference
        # currently resolves cache files with relative paths.
        atoms_path = audio_path.parent / f"{stem}_atoms.pt"
        contexts_path = audio_path.parent / f"{stem}_contexts.pt"
        atoms_cwd_path = cwd / f"{stem}_atoms.pt"
        contexts_cwd_path = cwd / f"{stem}_contexts.pt"

        # 5. Move atoms and contexts to CPU for saving
        atoms_cpu = [atom.detach().cpu() for atom in atoms]
        contexts_cpu = [ctx.detach().cpu() for ctx in contexts]

        # 6. Save atom and context tensors
        torch.save(atoms_cpu, atoms_path)
        torch.save(contexts_cpu, contexts_path)
        if atoms_cwd_path != atoms_path:
            torch.save(atoms_cpu, atoms_cwd_path)
            torch.save(contexts_cpu, contexts_cwd_path)

        # 7. Log cache location
        print(f"Saved cache for {audio_enum} -> {atoms_path.name}, {contexts_path.name}")
        print(f"  assets directory: {audio_path.parent}")
        if atoms_cwd_path != atoms_path:
            print(f"  cwd mirror directory: {cwd}")


if __name__ == "__main__":
    precompute()
