import torch
from pathlib import Path

from inference.methods import get_inference_engine
from inference.constants import AUDIO_ASSET_MAP


def precompute():
    engine = get_inference_engine()

    for audio_enum, audio_path in AUDIO_ASSET_MAP.items():
        audio_path = Path(audio_path).resolve()
        stem = audio_path.stem

        print(f"\nProcessing: {audio_enum}")

        # 1. Load audio
        audio_tensor = engine.load_audio_to_tensor(str(audio_path))

        # 2. Encode into atoms
        atoms = engine.encode_audio_to_atoms(audio_tensor)

        # 3. Compute contexts
        contexts = engine.compute_context_track(atoms)

        # 4. Set output cache paths
        atoms_path = audio_path.parent / f"{stem}_atoms.pt"
        contexts_path = audio_path.parent / f"{stem}_contexts.pt"

        # 5. Move atoms and contexts to CPU for saving
        atoms_cpu = [atom.detach().cpu() for atom in atoms]
        contexts_cpu = [ctx.detach().cpu() for ctx in contexts]

        # 6. Save atom and context tensors
        torch.save(atoms_cpu, atoms_path)
        torch.save(contexts_cpu, contexts_path)

        # 7. Log cache location
        print(f"Saved cache for {audio_enum} -> {atoms_path.name}, {contexts_path.name}")
        print(f"  directory: {audio_path.parent}")


if __name__ == "__main__":
    precompute()
