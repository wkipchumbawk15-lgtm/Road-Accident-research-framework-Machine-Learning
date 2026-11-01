import json
from pathlib import Path

MODEL_DIR = Path('/models')
FEATURE_ORDER_PATH = MODEL_DIR / 'feature_order.json'

if FEATURE_ORDER_PATH.exists():
    meta = json.loads(FEATURE_ORDER_PATH.read_text())
    # Support two formats for feature_order.json:
    # - a dict with keys 'predictors' and optional 'target'
    # - a simple list of predictor names
    if isinstance(meta, dict):
        FEATURE_ORDER = meta.get('predictors', [])
        TARGET_COLUMN = meta.get('target')
    elif isinstance(meta, list):
        FEATURE_ORDER = meta
        TARGET_COLUMN = None
    else:
        FEATURE_ORDER = []
        TARGET_COLUMN = None
else:
    FEATURE_ORDER = []
    TARGET_COLUMN = None

LABELS = {}
