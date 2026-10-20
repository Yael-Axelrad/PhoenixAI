"""Download quantized E5 model during Render build, not on a live request."""
from pathlib import Path
from huggingface_hub import hf_hub_download

root = Path(__file__).resolve().parents[1]
model_dir = root / "models" / "e5-onnx"
for filename in ("onnx/model_quantized.onnx", "tokenizer.json"):
    path = hf_hub_download(
        repo_id="Xenova/multilingual-e5-small",
        filename=filename,
        local_dir=str(model_dir),
    )
    print(f"Downloaded {filename}: {Path(path).stat().st_size / 1024**2:.1f} MB", flush=True)
print("E5 ONNX model ready", flush=True)
