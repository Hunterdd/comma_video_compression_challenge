import sys

class ModelHook:
    def find_spec(self, fullname, path, target=None):
        if fullname == "model":
            sys.meta_path.remove(self)
            try:
                import importlib.util
                spec = importlib.util.find_spec(fullname, path)
                if spec is not None:
                    orig_loader = spec.loader
                    class PatchedLoader:
                        def create_module(self, spec):
                            return orig_loader.create_module(spec)
                        def exec_module(self, module):
                            orig_loader.exec_module(module)
                            if hasattr(module, "HNeRVDecoder_grouped"):
                                module.HNeRVDecoder = module.HNeRVDecoder_grouped
                                print("[Hook] Successfully patched HNeRVDecoder to HNeRVDecoder_grouped", flush=True)
                    spec.loader = PatchedLoader()
                return spec
            finally:
                sys.meta_path.insert(0, self)
        return None

sys.meta_path.insert(0, ModelHook())
