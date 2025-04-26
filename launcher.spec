# -*- mode: python ; coding: utf-8 -*-

import os

LIB_PATH = os.path.join(".venv", "Lib", "site-packages")

a = Analysis(
    ['launcher.py'],
    pathex=[],
    datas=[
        (os.path.join(LIB_PATH, "en_core_web_trf"), "en_core_web_trf"),
        (os.path.join(LIB_PATH, "en_core_web_trf-3.8.0.dist-info"), "en_core_web_trf-3.8.0.dist-info"),
        (os.path.join(LIB_PATH, "en_core_web_sm"), "en_core_web_sm"),
        (os.path.join(LIB_PATH, "en_core_web_sm-3.8.0.dist-info"), "en_core_web_sm-3.8.0.dist-info"),
        (os.path.join(LIB_PATH, "curated_tokenizers"), "curated_tokenizers"),
        (os.path.join(LIB_PATH, "curated_tokenizers-0.0.9.dist-info"), "curated_tokenizers-0.0.9.dist-info"),
        (os.path.join(LIB_PATH, "curated_transformers"), "curated_transformers"),
        (os.path.join(LIB_PATH, "curated_transformers-0.1.1.dist-info"), "curated_transformers-0.1.1.dist-info"),
        (os.path.join(LIB_PATH, "spacy_transformers"), "spacy_transformers"),
        (os.path.join(LIB_PATH, "spacy_transformers-1.3.8.dist-info"), "spacy_transformers-1.3.8.dist-info"),
        (os.path.join(LIB_PATH, "spacy_curated_transformers"), "spacy_curated_transformers"),
        (os.path.join(LIB_PATH, "spacy_curated_transformers-0.3.0.dist-info"), "spacy_curated_transformers-0.3.0.dist-info"),
        (os.path.join(LIB_PATH, "spacy_alignments"), "spacy_alignments"),
        (os.path.join(LIB_PATH, "spacy_alignments-0.9.1.dist-info"), "spacy_alignments-0.9.1.dist-info"),
        (os.path.join(LIB_PATH, "spacy_lookups_data"), "spacy_lookups_data"),
        (os.path.join(LIB_PATH, "spacy_lookups_data-1.0.5.dist-info"), "spacy_lookups_data-1.0.5.dist-info"),

    ],
    binaries=[],
    hiddenimports=["spacy"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["ipython"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='launcher',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='launcher',
)
