import os
import subprocess
from pathlib import Path


os.chdir(Path(__file__).parent)

subprocess.run(["python", "app/scripts/app.py"])