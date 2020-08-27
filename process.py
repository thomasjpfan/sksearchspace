from pathlib import Path
from ConfigSpace.read_and_write import json as cs_json
from ConfigSpace.read_and_write import pcs_new

space_path = Path("sksearchspace")

for path in space_path.glob("**/*.pcs_new"):
    path.unlink()
