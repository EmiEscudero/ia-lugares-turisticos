"""Community Cloud entrypoint; the shared application owns model initialization."""
import os
import runpy
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
os.environ["TOURIST_HOSTED"] = "1"
os.environ["TOURIST_MODEL_DIR"] = str(PROJECT / "artifacts" / "hosted_models")
runpy.run_path(str(PROJECT / "app.py"), run_name="__main__")
