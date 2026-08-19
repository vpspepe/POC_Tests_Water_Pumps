import subprocess
from os.path import join as pjoin

subprocess.run(
    [
        "git",
        "clone",
        "https://github.com/vpspepe/smart.git",
        pjoin(".", "pump2d_smart", "smart"),
    ],
    check=True,
)
