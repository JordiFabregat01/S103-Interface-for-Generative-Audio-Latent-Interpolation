import torch
from inference.constants import AUDIO_ASSET_MAP, AUDIO_CACHE_MAP, CACHE_DIR
from inference.scapes_runtime import EncodecProcessor
from inference.methods import CLAPWrapper  # ONLY if you need context embeddings

import os


def precompute():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # ONLY lightweight components (NO FlowInference)
    processor = EncodecProcessor(sr=48000, streamable=True, device=device)

    # optional: only needed for contexts
    context_model = CLAPWrapper(version="2023", use_cuda=(device == "cuda"))

    for audio_enum, path in AUDIO_ASSET_MAP.items():

        print(f"\nProcessing {audio_enum}...")

        # 1. load audio
        audio, _ = processor.load_audio_to_tensor(str(path))

        # 2. encode → atoms (THIS is your real goal)
        latent_list, metadata = processor.audio_to_latents(audio, sr=48000)

        latent = torch.cat(latent_list, dim=-1)

        scale = metadata["audio_scales"][0]
        scale = scale.unsqueeze(-1).expand(-1, -1, latent.shape[-1])

        atoms = torch.cat([latent, scale], dim=1)

        # 3. compute contexts (optional but consistent with your system)
        contexts = []

        hop = atoms.shape[-1] // 10  # simple sliding approximation

        for i in range(10):
            start = i * hop
            end = start + hop

            segment = audio[:, :, start * 320:(end + 1) * 320]

            emb = context_model.compute_embedding(
                segment,
                og_sr=48000,
                random_extension=False
            ).squeeze(0)

            contexts.append(emb)

        # 4. save
        name = AUDIO_CACHE_MAP[audio_enum]
        save_dir = CACHE_DIR / name
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(atoms, save_dir / "atoms.pt")
        torch.save(contexts, save_dir / "contexts.pt")

        print(f"Saved: {name}")


if __name__ == "__main__":
    precompute()