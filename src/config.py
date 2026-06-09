from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

# FAERS Q1 2026 filenames
FAERS_DEMO = DATA_RAW / "DEMO26Q1.txt"
FAERS_DRUG = DATA_RAW / "DRUG26Q1.txt"
FAERS_REAC = DATA_RAW / "REAC26Q1.txt"
FAERS_INDI = DATA_RAW / "INDI26Q1.txt"
FAERS_OUTC = DATA_RAW / "OUTC26Q1.txt"

# Sampling
SAMPLE_SIZE = 8000
RANDOM_SEED = 42
TEST_SIZE = 0.30

# Model
BIOBERT_MODEL = "dmis-lab/biobert-base-cased-v1.2"
DEVICE = "cuda"
