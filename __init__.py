import os

try:
    import folder_paths
except ModuleNotFoundError as exc:
    if exc.name != "folder_paths":
        raise
    # Allow metadata tooling and the standalone test suite to import the
    # package outside a ComfyUI installation.
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
else:
    # Register a dedicated subfolder: ComfyUI/models/MelBandRoFormer/
    _model_dir = os.path.join(folder_paths.models_dir, "MelBandRoFormer")
    os.makedirs(_model_dir, exist_ok=True)
    folder_paths.add_model_folder_path("MelBandRoFormer", _model_dir)

    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
