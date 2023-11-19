import torch
from safetensors.torch import save_file
from pathlib import Path
import pickle
from types import SimpleNamespace

# Change this path to the path of your model
MODEL_PATH = Path(r"./path/to/model.pth")


safe_list = {
    ("collections", "OrderedDict"),
    ("typing", "OrderedDict"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch", "BFloat16Storage"),
    ("torch", "FloatStorage"),
    ("torch", "HalfStorage"),
    ("torch", "IntStorage"),
    ("torch", "LongStorage"),
    ("torch", "DoubleStorage"),
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        # Only allow required classes to load state dict
        if (module, name) not in safe_list:
            raise pickle.UnpicklingError(f"Global '{module}.{name}' is forbidden")
        return super().find_class(module, name)


RestrictedUnpickle = SimpleNamespace(
    Unpickler=RestrictedUnpickler,
    __name__="pickle",
    load=lambda *args, **kwargs: RestrictedUnpickler(*args, **kwargs).load(),
)


model = torch.load(MODEL_PATH, pickle_module=RestrictedUnpickle)

# Supports extracting the tensors out of models trained with basicsr
if "params_ema" in model:
    model = model["params_ema"]
elif "params" in model:
    model = model["params"]


# Save the model as a SafeTensors file
save_file(model, f"./{MODEL_PATH.name}.safetensors")
