import argparse
import torch
from strhub.models.utils import load_from_checkpoint
from torch.utils.mobile_optimizer import optimize_for_mobile
import os

parser = argparse.ArgumentParser()
parser.add_argument("--source_path", default=r"D:\ACG\parseq\epoch=23-step=5464223-val_accuracy=91.6810-val_NED=97.0141.ckpt", help="Path to the trained model ckpt file.")
parser.add_argument("--target_path", default="D:/ACG/parseq/", help="Path where to export the trained model.")
parser.add_argument("--optimize_for_mobile", action='store_true', help="Whether to apply optimization for mobile.")
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

# Quantize
quantized_parseq = torch.quantization.quantize_dynamic(parseq, {torch.nn.Linear}, dtype=torch.qint8)

# Create dummy input
dummy_tensor = torch.rand((1, 3, parseq.hparams.img_size[0], parseq.hparams.img_size[1]))

# Trace
torchscript_model = torch.jit.trace(quantized_parseq, dummy_tensor)

# Optimize for mobile
if args.optimize_for_mobile:
    torchscript_model = optimize_for_mobile(torchscript_model)

# Save
os.makedirs(args.target_path, exist_ok=True)
target_file = os.path.join(args.target_path, "parseq_optimized.pt")
torch.jit.save(torchscript_model, target_file)

print(f"✅ Saved optimized model to: {target_file}")