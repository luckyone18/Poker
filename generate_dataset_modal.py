"""
Modal job: Generate Behavioral Cloning dataset from expert bots.

Usage:
    modal run generate_dataset_modal.py --expert smartbot --episodes 2000
    modal run generate_dataset_modal.py --expert mc200 --episodes 1000
"""
import modal
import os
import sys

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=1.9.0")
    .pip_install("numpy")
    .add_local_dir(
        ".",
        remote_path="/root/poker",
        ignore=["__pycache__", "*.pyc", ".git", "venv", ".venv", "output", ".mypy_cache", "models"],
    )
)

volume = modal.Volume.from_name("poker-models", create_if_missing=True)
app = modal.App("bc-dataset-gen", image=image)


@app.function(
    volumes={"/root/models": volume},
    timeout=3600,
)
def generate_smartbot(episodes: int = 2000):
    """Generate BC dataset from SmartBot expert."""
    sys.path.insert(0, "/root/poker")

    # Symlink hack
    poker_models = "/root/poker/models"
    volume_models = "/root/models"
    if os.path.exists(poker_models) and not os.path.islink(poker_models):
        import shutil
        shutil.rmtree(poker_models)
    if not os.path.exists(poker_models):
        os.symlink(volume_models, poker_models)

    from training.generate_bc_dataset import generate_dataset
    generate_dataset(
        expert_type="smartbot",
        num_episodes=episodes,
        chips=2000,
        output_path="/root/models/bc_smartbot.pkl",
    )
    print("✅ SmartBot dataset saved to volume")


@app.function(
    volumes={"/root/models": volume},
    timeout=3600,
)
def generate_mc(episodes: int = 1000):
    """Generate BC dataset from MC(200) expert."""
    sys.path.insert(0, "/root/poker")

    poker_models = "/root/poker/models"
    volume_models = "/root/models"
    if os.path.exists(poker_models) and not os.path.islink(poker_models):
        import shutil
        shutil.rmtree(poker_models)
    if not os.path.exists(poker_models):
        os.symlink(volume_models, poker_models)

    from training.generate_bc_dataset import generate_dataset
    generate_dataset(
        expert_type="mc200",
        num_episodes=episodes,
        chips=2000,
        output_path="/root/models/bc_mc200.pkl",
    )
    print("✅ MC200 dataset saved to volume")


@app.local_entrypoint()
def main(expert: str = "smartbot", episodes: int = 2000):
    if expert == "smartbot":
        generate_smartbot.remote(episodes=episodes)
    elif expert == "mc200":
        generate_mc.remote(episodes=episodes)
    else:
        print(f"Unknown expert: {expert}")