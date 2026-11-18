# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Compatibility shim for remote Molmo implementations on newer Transformers."""

from contextlib import contextmanager
from functools import wraps
from inspect import signature
from pathlib import Path
from threading import RLock

_COMPAT_LOCK = RLock()
_PATCH_MARKER = "_qeff_legacy_molmo_transformers_compat"
_IMPORTS_PATCH_MARKER = "_qeff_legacy_molmo_optional_import_compat"


def is_molmo_model_reference(pretrained_model_name_or_path, config=None):
    """Return whether a checkpoint reference or supplied config identifies Molmo."""
    if isinstance(config, dict):
        model_type = config.get("model_type")
        architectures = config.get("architectures") or []
    else:
        model_type = getattr(config, "model_type", None)
        architectures = getattr(config, "architectures", []) or []
    if isinstance(architectures, str):
        architectures = [architectures]

    return (
        model_type == "molmo"
        or "MolmoForCausalLM" in architectures
        or "molmo" in str(pretrained_model_name_or_path).lower()
    )


def _filter_molmo_optional_imports(filename, imports):
    if Path(filename).name == "image_preprocessing_molmo.py":
        # Transformers scans function-local imports as hard requirements, though
        # Molmo defaults to torch-bilinear and uses TF only for an optional path.
        return [module for module in imports if module != "tensorflow"]
    return imports


def _wrap_molmo_import_scanner(get_imports):
    if getattr(get_imports, _IMPORTS_PATCH_MARKER, False):
        return get_imports

    @wraps(get_imports)
    def get_imports_with_optional_tensorflow(filename, *args, **kwargs):
        imports = get_imports(filename, *args, **kwargs)
        return _filter_molmo_optional_imports(filename, imports)

    setattr(get_imports_with_optional_tensorflow, _IMPORTS_PATCH_MARKER, True)
    return get_imports_with_optional_tensorflow


def _patch_legacy_molmo_class(model_class):
    if getattr(model_class, "__name__", None) != "MolmoForCausalLM":
        return model_class
    if getattr(model_class, _PATCH_MARKER, False):
        return model_class

    from transformers import PreTrainedModel

    if not isinstance(model_class, type) or not issubclass(model_class, PreTrainedModel):
        return model_class

    original_init = model_class.__init__
    original_tie_weights = getattr(model_class, "tie_weights", None)
    if callable(original_tie_weights):
        try:
            tie_parameters = signature(original_tie_weights).parameters
        except (TypeError, ValueError) as error:
            raise RuntimeError("Could not inspect MolmoForCausalLM.tie_weights() for compatibility.") from error

        # This only adapts the legacy no-argument override. Newer Transformers
        # passes these kwargs during loading; old remote Molmo does not accept them.
        if list(tie_parameters) == ["self"]:

            @wraps(original_tie_weights)
            def tie_weights_compat(self, *args, missing_keys=None, recompute_mapping=False, **kwargs):
                if args or kwargs:
                    raise TypeError(
                        "Legacy Molmo tie_weights() received unsupported arguments: "
                        f"args={args}, kwargs={tuple(kwargs)}"
                    )
                return original_tie_weights(self)

            model_class.tie_weights = tie_weights_compat

    @wraps(original_init)
    def init_with_post_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, "all_tied_weights_keys"):
            # Safe only because this patch targets legacy Molmo, whose remote
            # implementation omitted PreTrainedModel.post_init() entirely. Do not
            # apply this lifecycle workaround to arbitrary remote model classes.
            self.post_init()
        if not hasattr(self, "all_tied_weights_keys"):
            raise RuntimeError("Molmo post_init() did not initialize all_tied_weights_keys.")

    model_class.__init__ = init_with_post_init
    setattr(model_class, _PATCH_MARKER, True)
    return model_class


@contextmanager
def legacy_remote_molmo_compat():
    """Temporarily adapt remote Molmo loading to Transformers 5.

    The lock serializes uses of this shim, but Transformers' patched hooks remain
    process-global and may be observed by unrelated threads during this context.
    """
    from transformers import dynamic_module_utils
    from transformers.models.auto import auto_factory

    with _COMPAT_LOCK:
        original_loader = auto_factory.get_class_from_dynamic_module
        original_get_imports = dynamic_module_utils.get_imports
        imports_wrapper = None

        @wraps(original_loader)
        def loader_with_molmo_compat(*args, **kwargs):
            return _patch_legacy_molmo_class(original_loader(*args, **kwargs))

        if not getattr(original_get_imports, _IMPORTS_PATCH_MARKER, False):
            imports_wrapper = _wrap_molmo_import_scanner(original_get_imports)
            dynamic_module_utils.get_imports = imports_wrapper

        auto_factory.get_class_from_dynamic_module = loader_with_molmo_compat
        try:
            yield
        finally:
            if auto_factory.get_class_from_dynamic_module is loader_with_molmo_compat:
                auto_factory.get_class_from_dynamic_module = original_loader
            if imports_wrapper is not None and dynamic_module_utils.get_imports is imports_wrapper:
                dynamic_module_utils.get_imports = original_get_imports


def install_legacy_molmo_processor_import_compat():
    """Install the process-wide Molmo optional-import filter.

    This is installed when QEfficient is imported, so direct calls to
    Transformers' ``AutoProcessor.from_pretrained()`` work even when the
    processor is loaded before the QEff model. The filename-filtered patch
    remains active for the process lifetime; repeated calls are idempotent. It
    only removes TensorFlow from dependency scanning for
    ``image_preprocessing_molmo.py``. Selecting the optional TensorFlow resize
    method still requires TensorFlow to be installed.
    """
    from transformers import dynamic_module_utils

    with _COMPAT_LOCK:
        dynamic_module_utils.get_imports = _wrap_molmo_import_scanner(dynamic_module_utils.get_imports)
