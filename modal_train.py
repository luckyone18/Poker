"""
Modal wrapper for Poker RL training with graded curriculum.

Usage (after `modal token set`):
    # Test imports (free, CPU-only)
    modal run modal_train.py::quick_test

    # Run training (5k episodes = ~2h on T4, good first test)
    modal run modal_train.py --episodes 5000

    # Resume from checkpoint
    modal run modal_train.py --episodes 5000 --resume models/rl_model_omc.pt

    # Download model
    modal volume get poker-models rl_model_omc_final.pt ~/Poker/models/

Checkpoint strategy:
    - Auto-saved every 500 episodes to /root/models/rl_model_omc.pt
    - Final model: /root/models/rl_model_omc_final.pt
    - CSV log: /root/models/rl_omc_training_log.csv (for downloading later)
"""

import modal
import os
import sys

# ── Image with all dependencies + source code ─────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=1.9.0", "matplotlib>=3.5.0", "treys>=0.1.8")
    .pip_install("numpy")
    .add_local_dir(
        ".",
        remote_path="/root/poker",
        ignore=["__pycache__", "*.pyc", ".git", "venv", ".venv", "output", ".mypy_cache", "models", "*.sock", ".pm2"],
    )
)

# ── Named volume for persistent storage ───────────────────────────────────
volume = modal.Volume.from_name("poker-models", create_if_missing=True)

app = modal.App("poker-training", image=image)


@app.cls(
    gpu="T4",
    volumes={"/root/models": volume},
    timeout=3600 * 4,            # 4 hour max
)
class Trainer:
    @modal.enter()
    def setup(self):
        """Set up project path and import training module."""
        sys.path.insert(0, "/root/poker")
        print(f"✅ Python {sys.version}")
        print(f"✅ PyTorch {__import__('torch').__version__}")
        print(f"✅ CUDA available: {__import__('torch').cuda.is_available()}")

    @modal.method()
    def train(
        self,
        episodes: int = 5_000,
        chips: int = 2000,
        lr_step: int = 25_000,
        resume: str = None,
    ):
        """Run the RL self-play training with graded curriculum."""
        from training.train_rl_omc import train_rl_bot

        print(f"🚀 Starting graded curriculum training:")
        print(f"   Episodes: {episodes}")
        print(f"   Chips/player: {chips}")
        print(f"   LR step every: {lr_step}")
        print(f"   Resume: {resume or 'N/A'}")

        # Ensure output directories
        os.makedirs("/root/output", exist_ok=True)
        os.makedirs("/root/models", exist_ok=True)

        # HACK: symlink volume for persistent models
        poker_models = "/root/poker/models"
        volume_models = "/root/models"
        if os.path.exists(poker_models) and not os.path.islink(poker_models):
            import shutil
            shutil.rmtree(poker_models)
        if not os.path.exists(poker_models):
            os.symlink(volume_models, poker_models)

        # Resolve resume path
        resume_path = None
        if resume:
            resume_path = os.path.join("/root/poker", resume) if not resume.startswith("/") else resume

        train_rl_bot(
            num_episodes=episodes,
            chips_per_player=chips,
            csv_path="/root/models/rl_omc_training_log.csv",
            lr_step_episodes=lr_step,
            resume_from=resume_path,
        )

        print("✅ Training complete!")
        print(f"   Model: /root/models/rl_model_omc_final.pt")
        print(f"   CSV:   /root/models/rl_omc_training_log.csv")
        # Volume commit happens automatically on exit


@app.local_entrypoint()
def main(
    episodes: int = 5_000,
    chips: int = 500,
    lr_step: int = 25_000,
    resume: str = None,
):
    """Local entry point — called from `modal run`."""
    trainer = Trainer()
    trainer.train.remote(
        episodes=episodes,
        chips=chips,
        lr_step=lr_step,
        resume=resume,
    )


# ── Quick test (CPU, no GPU) ─────────────────────────────────────────────
@app.function()
def quick_test():
    """Quick sanity test — verify all imports work."""
    sys.path.insert(0, "/root/poker")

    from core.engine import Table
    t = Table()
    print(f"✅ Engine OK — Table created")

    from bots.rl_bot import RLBot
    rl = RLBot(model_path="", training_mode=True)
    print(f"✅ RLBot OK — device={rl.device}")

    from bots.poker_mind_bot import SmartBot
    sb = SmartBot()
    print(f"✅ SmartBot OK — {type(sb).__name__}")

    from bots.monte_carlo_bot import MonteCarloBot as MCB
    mc = MCB(simulations=50)
    print(f"✅ MonteCarloBot OK — {type(mc).__name__}")

    print("✅ ALL IMPORTS PASSED!")