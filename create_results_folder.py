from pathlib import Path

# directory base (dove si trova questo script)
BASE_DIR = Path(__file__).resolve().parent

results_dir = BASE_DIR / "results"
results_dir.mkdir(exist_ok=True)

for i in range(1, 8):
    img_dir = results_dir / f"test{i}" / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

print("Struttura creata (o già esistente).")
