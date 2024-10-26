# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('my_yolov8m.pt','.'),('my_yolov8m.onnx','.'),('C_SQUARE_ENIX.png','.'),('WoLNamesBlackedOut.ico','.'),],
    hiddenimports=[],
    hookspath=['./hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

splash = Splash('WoLNamesBlackedOut.png',binaries=a.binaries,datas=a.datas,)
exe = EXE(
    pyz,
    splash,                   # <-- splash target
    a.scripts,
    [],
    exclude_binaries=True,
    name='WoLNamesBlackedOut',
    debug=False,     #
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,   #
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='.\WoLNamesBlackedOut.ico',
)
coll = COLLECT(
    exe,
    splash.binaries,     # <-- splash binaries
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='WoLNamesBlackedOut',
)
