import argparse
import torch
import torch.onnx
from strhub.models.utils import load_from_checkpoint
import os

parser = argparse.ArgumentParser()
parser.add_argument("--source_path", default=r"D:\ACG\parseq\epoch=23-step=5464223-val_accuracy=91.6810-val_NED=97.0141.ckpt", help="Path to the trained model ckpt file.")
parser.add_argument("--target_path", default="D:/ACG/parseq/", help="Path where to export the trained model.")
parser.add_argument("--quantize", action='store_true', help="Whether to apply dynamic quantization before export.")
parser.add_argument("--opset_version", type=int, default=17, help="ONNX opset version (default: 11)")
args = parser.parse_args()

# Load the model
parseq = load_from_checkpoint(args.source_path)
parseq.eval()

# ⚠️ Safely remove Lightning hooks and attributes
# Use __dict__ to avoid triggering property getters
if '_trainer' in parseq.__dict__:
    delattr(parseq, '_trainer')

# Check for trainer in the class dict to avoid property access
if 'trainer' in parseq.__class__.__dict__:
    # Remove the property descriptor if it exists
    try:
        del parseq.__class__.__dict__['trainer']
    except (KeyError, TypeError):
        pass

# Alternative approach: set _trainer to None if it exists in __dict__
if hasattr(parseq, '__dict__') and '_trainer' in parseq.__dict__:
    parseq._trainer = None

# Remove other Lightning-specific methods that cause issues during tracing
lightning_methods = ['training_step', 'validation_step', 'configure_optimizers',
                    'on_train_start', 'on_validation_start', 'on_test_start']

for attr in lightning_methods:
    if hasattr(parseq, attr):
        try:
            delattr(parseq, attr)
        except AttributeError:
            pass

# Optional quantization
if args.quantize:
    print("Applying dynamic quantization...")
    parseq = torch.quantization.quantize_dynamic(parseq, {torch.nn.Linear}, dtype=torch.qint8)

# Create dummy input
dummy_tensor = torch.rand((1, 3, parseq.hparams.img_size[0], parseq.hparams.img_size[1]))

# Prepare output path
os.makedirs(args.target_path, exist_ok=True)
target_file = os.path.join(args.target_path, "parseq_model_quan.onnx")

# Export to ONNX
print("Exporting model to ONNX...")
torch.onnx.export(
    parseq,                          # Model to export
    dummy_tensor,                    # Model input (or tuple for multiple inputs)
    target_file,                     # Where to save the model
    export_params=True,              # Store the trained parameter weights inside the model file
    opset_version=args.opset_version, # ONNX version to export to
    do_constant_folding=True,        # Whether to execute constant folding for optimization
    input_names=['input'],           # Model's input names
    output_names=['output'],         # Model's output names
    dynamic_axes={
        'input': {0: 'batch_size'},  # Variable batch size
        'output': {0: 'batch_size'}
    }
)

print(f"✅ Saved ONNX model to: {target_file}")

# Optional: Verify the ONNX model
try:
    import onnx
    onnx_model = onnx.load(target_file)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model verification passed!")
except ImportError:
    print("⚠️ Install 'onnx' package to verify the exported model")
except Exception as e:
    print(f"⚠️ ONNX model verification failed: {e}")