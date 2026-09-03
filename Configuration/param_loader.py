"""
Preset-based parameter loading for algorithm-side configuration.

Layering (later layers win):
  1. Class defaults    — Configuration/{ppo,a2c,idql,qmix}_params.py
  2. Task preset JSON  — Configuration/presets/{family}_{task}.json (auto-loaded)
  3. User config JSON  — passed via --config (sparse, only changed fields)

Presets and user configs are sparse: write only the fields that differ from
the layer below. Derived fields (see DERIVED_FIELDS on each params class)
are recomputed by _derive() after merging and cannot be set from JSON.

Task-implied constraints (auto-applied, not user preferences):
  - POSIG + A2C family requires no_sharing=False; explicitly setting it to
    True raises an error.
"""

import json
import os

ALGO_FAMILY = {
    "ia2c": "a2c",
    "maa2c": "a2c",
    "ippo": "ppo",
    "mappo": "ppo",
    "idql": "idql",
    "hys": "idql",
    "vdn": "qmix",
    "qmix": "qmix",
}

PRESET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def task_key_for(env_name, loc):
    """Map (env_name, loc) to a preset task key: NFIG / SIG_SL / SIG_ML / POSIG."""
    if env_name == "NFIG":
        if loc is None:
            raise ValueError("NFIG requires --loc.")
        return "NFIG"
    if env_name == "SIG":
        return "SIG_SL" if loc is not None else "SIG_ML"
    if env_name == "POSIG":
        if loc is not None:
            raise ValueError("POSIG must be run with --loc omitted.")
        return "POSIG"
    raise ValueError(f"Unknown env: {env_name}")


def _load_json(path, what):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{what} file not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not all(isinstance(k, str) for k in data):
        raise ValueError(f"{what} file must be a flat JSON object mapping parameter names to values: {path}")
    return data


def resolve_param_overrides(algo, env_name, loc, user_config_path=None):
    """
    Returns (overrides, preset_path, user_config_path):
      overrides         merged dict to apply on top of class defaults
      preset_path       auto-loaded preset path, or None if no preset exists
      user_config_path  echoed back for logging
    """
    if algo not in ALGO_FAMILY:
        raise ValueError(f"Unknown algo: {algo}. Use one of {sorted(ALGO_FAMILY)}")
    family = ALGO_FAMILY[algo]
    task_key = task_key_for(env_name, loc)

    overrides = {}
    preset_path = os.path.join(PRESET_DIR, f"{family}_{task_key}.json")
    loaded_preset = preset_path if os.path.isfile(preset_path) else None
    if loaded_preset:
        overrides.update(_load_json(loaded_preset, "Preset"))
        loaded_preset = "./" + os.path.relpath(loaded_preset, _REPO_ROOT)
    if user_config_path:
        overrides.update(_load_json(user_config_path, "User config"))

    if task_key == "POSIG" and family == "a2c":
        if overrides.get("no_sharing") is True:
            raise ValueError("POSIG requires parameter sharing: 'no_sharing' must be false.")
        overrides["no_sharing"] = False

    return overrides, loaded_preset, user_config_path


def apply_overrides(params_obj, overrides):
    """Validate and apply an overrides dict onto a params instance, then recompute derived fields."""
    if not overrides:
        return
    derived = getattr(type(params_obj), "DERIVED_FIELDS", ())
    for key, value in overrides.items():
        if key in derived:
            raise ValueError(
                f"'{key}' is a derived field of {type(params_obj).__name__} "
                f"(recomputed from other parameters); remove it from the JSON."
            )
        if not hasattr(params_obj, key):
            raise ValueError(f"Unknown parameter '{key}' for {type(params_obj).__name__} (typo?)")
        setattr(params_obj, key, value)
    if hasattr(params_obj, "_derive"):
        params_obj._derive()
