"""pump2d_smart package initialization.

Ensures that pump_static_rom is importable across environments,
falling back to local repository path resolution if not installed in site-packages.
"""

import sys
from pathlib import Path

# Ensure pump_static_rom is importable
try:
    import pump_static_rom
except ImportError:
    curr_dir = Path(__file__).resolve().parent
    psr_src = (curr_dir.parent.parent / "PumpStaticROM" / "src").resolve()
    if psr_src.exists() and str(psr_src) not in sys.path:
        sys.path.insert(0, str(psr_src))
    try:
        import pump_static_rom
    except ImportError as e:
        print(f"Warning: Could not import pump_static_rom from {psr_src}: {e}")
