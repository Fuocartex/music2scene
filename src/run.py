
from pathlib import Path
import shutil
import subprocess
import sys
import time

dirs = [
    "live_slices",
    "live_frames",
    "live_interp"
]

for d in dirs:
    path = Path(d)
    if path.exists():
        print(f" Pulisco {d}")
        shutil.rmtree(path)
        path.mkdir()  # ricrea cartella vuota

print(" Pulizia completata")


processes = []

commands = [
    "python src\live_show.py",
    "python src\inf1.py", 
    "python src\live_slicing.py",     
                   
]

print(" Avvio pipeline completa...\n")

for cmd in commands:
    print(f" Avvio: {' '.join(cmd)}")
    subprocess.Popen(["cmd", "/k", cmd], creationflags=subprocess.CREATE_NEW_CONSOLE)
    #processes.append(p)
    if cmd == "python src\inf1.py":
        time.sleep(20)
    else:
        time.sleep(2)

print("\n Tutti i processi avviati")

try:
    for p in processes:
        p.wait()
except KeyboardInterrupt:
    print("\n Stop manuale... chiudo tutto")

    for p in processes:
        p.terminate()

    sys.exit()