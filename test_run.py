"""
Test run stage4_run.train() directly via Python API.
This should give us a proper traceback if the function fails.
"""
import sys, traceback
sys.path.insert(0, "/root/Poker")

import modal

# Get the app
app = modal.App.lookup("poker-stage4")
print(f"App found: {app}")

# Get the train function from the app
train_fn = app._lookup("train")
print(f"train function found: {train_fn}")

print("Calling train(2, 2000, 3e-5, 512) with timeout=300s...")
try:
    result = train_fn.call(2, 2000, 3e-5, 512, timeout=300)
    print(f"Result: {result}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()