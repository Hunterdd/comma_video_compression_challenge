import sys

class ModelHook:
    def find_spec(self, fullname, path, target=None):
        if fullname == "model":
            # Remove ourselves to prevent infinite recursion
            sys.meta_path.remove(self)
            try:
                import importlib
                # Load the real module from the path
                mod = importlib.import_module("model")
                # Patch HNeRVDecoder with HNeRVDecoder_grouped if it exists
                if hasattr(mod, "HNeRVDecoder_grouped"):
                    mod.HNeRVDecoder = mod.HNeRVDecoder_grouped
                    print("[Hook] Successfully patched HNeRVDecoder to HNeRVDecoder_grouped", flush=True)
                return importlib.util.find_spec("model")
            finally:
                sys.meta_path.insert(0, self)
        return None

# Install the import hook
sys.meta_path.insert(0, ModelHook())
