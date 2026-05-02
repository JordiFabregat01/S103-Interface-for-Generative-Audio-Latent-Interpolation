import torch
from inference.methods import get_inference_engine
from inference.constants import AUDIO_ASSET_MAP, AUDIO_CACHE_MAP, CACHE_DIR


def precompute():
    engine = get_inference_engine()

    for audio_enum, audio_path in AUDIO_ASSET_MAP.items():

        print(f"\nProcessing: {audio_enum}")

        # 1. Load audio
        audio_tensor = engine.load_audio_to_tensor(str(audio_path))

        # 2. Encode into atoms
        atoms = engine.encode_audio_to_atoms(audio_tensor)

        # 3. Compute contexts
        contexts = engine.compute_context_track(atoms)

        # 4. Create folder
        name = AUDIO_CACHE_MAP[audio_enum]
        save_dir = CACHE_DIR / name
        save_dir.mkdir(parents=True, exist_ok=True)

        # 5. Save files
        torch.save(atoms, save_dir / "atoms.pt")
        torch.save(contexts, save_dir / "contexts.pt")

        print(f"Saved cache for {audio_enum} → {save_dir}")


if __name__ == "__main__":
    precompute()