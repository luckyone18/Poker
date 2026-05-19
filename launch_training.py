#!/usr/bin/env python3
"""
Launch poker RL training via Modal Python client.
This runs independently of terminal session.
"""
import modal

# Same image and volume as modal_train.py
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=1.9.0", "matplotlib>=3.5.0", "treys>=0.1.8")
    .pip_install("numpy")
)

volume = modal.Volume.from_name("poker-models", create_if_missing=True)
app = modal.App("poker-training-v2", image=image)

@app.cls(
    gpu="T4",
    volumes={"/root/models": volume},
    timeout=3600 * 4,
)
class Trainer:
    @modal.method()
    def train(self, episodes: int = 5000, resume: str = None):
        import os, sys
        sys.path.insert(0, "/root/poker")
        
        os.makedirs("/root/models", exist_ok=True)
        poker_models = "/root/poker/models"
        volume_models = "/root/models"
        if os.path.exists(poker_models) and not os.path.islink(poker_models):
            import shutil
            shutil.rmtree(poker_models)
        if not os.path.exists(poker_models):
            os.symlink(volume_models, poker_models)
        
        from training.train_rl_omc import train_rl_bot
        print(f"Starting RL training: {episodes} episodes, resume={resume}")
        
        resume_path = None
        if resume:
            resume_path = os.path.join("/root/poker", resume) if not resume.startswith("/") else resume
        
        train_rl_bot(
            num_episodes=episodes,
            chips_per_player=500,
            csv_path="/root/models/rl_run3_training_log.csv",
            lr_step_episodes=25000,
            resume_from=resume_path,
        )
        print("Training complete!")


if __name__ == "__main__":
    # Parse args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    print(f"Launching training: {args.episodes} episodes, resume={args.resume}")
    with app.run():
        Trainer().train.remote(args.episodes, args.resume)
        print("Done!")