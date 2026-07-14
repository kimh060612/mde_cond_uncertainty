from pathlib import Path as _Path
import os as _os
_SCRIPT_DIR = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
_os.chdir(_SCRIPT_DIR)
del _Path, _os

from utils.select_canonical import SelectCanonicalandMatchFrames

def main():
    return SelectCanonicalandMatchFrames.from_env().run()

if __name__ == "__main__":
    main()
