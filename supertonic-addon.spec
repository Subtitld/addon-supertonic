# PyInstaller spec for the supertonic add-on.
# Build with: pyinstaller supertonic-addon.spec --distpath dist/
# Resulting dist/supertonic-addon/ + manifest.json + LICENSE + README.md is
# zipped into supertonic-<version>-<platform>.zip for the catalog.
#
# Considerably lighter than the qwen3-tts / coqui-xtts specs: supertonic
# runs on onnxruntime directly with no torch / transformers dependency,
# so the frozen bundle weighs in around ~200 MB instead of ~3 GB.

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


def _safe_collect(fn, name):
    """Same defensive pattern as the other addons — never fail the freeze
    on an absent optional sub-package; runtime imports are what matter."""
    try:
        return fn(name)
    except Exception:
        return []


hiddenimports = (
    _safe_collect(collect_submodules, 'supertonic')
    # onnxruntime ships its EPs as separately-loaded native libraries;
    # collect_submodules picks up the Python shim. The C extensions live
    # under `onnxruntime.capi` and are pulled in transitively via binaries.
    + _safe_collect(collect_submodules, 'onnxruntime')
    + _safe_collect(collect_submodules, 'soundfile')
    + _safe_collect(collect_submodules, 'huggingface_hub')
)

datas = (
    # Supertonic package ships a small phonemizer table / config files
    # next to the .py modules — collect them all so the frozen binary
    # finds them via __file__ introspection at runtime. Match the
    # include_py_files=True pattern coqui-xtts uses for the same reason.
    _safe_collect(lambda n: collect_data_files(n, include_py_files=True), 'supertonic')
    + _safe_collect(collect_data_files, 'onnxruntime')
    + _safe_collect(collect_data_files, 'soundfile')
    + _safe_collect(collect_data_files, 'huggingface_hub')
    + [('manifest.json', '.')]
)

block_cipher = None

a = Analysis(
    ['supertonic_addon.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Aggressive exclusion list — none of these are pulled by supertonic
    # but they sometimes sneak in via build env (developers who also
    # have torch/transformers installed for other addons). Excluding
    # them up front keeps the bundle small even on shared build hosts.
    excludes=['tensorflow', 'jax', 'flax', 'gradio',
              'torch', 'torchaudio', 'torchvision',
              'transformers', 'accelerate', 'datasets',
              'scipy.spatial.cKDTree'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='supertonic-addon',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=False, upx_exclude=[],
    name='supertonic-addon',
)
