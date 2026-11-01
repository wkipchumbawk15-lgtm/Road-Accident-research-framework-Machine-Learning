from utils.train_model import train_all_models, load_dataset

print("[monitor] loading dataset", flush=True)
X,y = load_dataset()
print(f"[monitor] loaded rows={len(X)}", flush=True)
print("[monitor] starting train_all_models", flush=True)
train_all_models()
print("[monitor] train_all_models returned", flush=True)
