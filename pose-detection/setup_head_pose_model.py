"""Acquire pinned OMZ FP32 XML/BIN and verify official SHA384 checksums."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]

NAME = "head-pose-estimation-adas-0001"
BASE = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/{NAME}/FP32"
# Open Model Zoo 2023.3.0 model.yml (Apache-2.0).
FILES = {
    "xml": (53705,"7b155bf7821ff6e7f46e32fc6fc2cc11c7f8692957e3762d84e2aaba2718a5e07dfd1eb885a3f2953ac2b170d83975a7"),
    "bin": (7647616,"29c6f24561fd81516c2d12c3648fb8a5018b1a09846ef57a664667f39c3796b1007c71d8c8801eb0babaf38ac8c2556e"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,default=REPO_ROOT/"models/head_pose")
    args = parser.parse_args()
    args.directory.mkdir(parents=True,exist_ok=True)
    for extension,(size,digest) in FILES.items():
        target = args.directory/f"{NAME}.{extension}"
        if target.exists():
            data = target.read_bytes()
            if len(data)==size and hashlib.sha384(data).hexdigest()==digest:
                print(f"Verified {target}")
                continue
            raise RuntimeError(f"Existing model checksum mismatch: {target}; move it aside before retrying")
        with urllib.request.urlopen(f"{BASE}/{target.name}",timeout=30) as response:
            data = response.read(size+1)
        if len(data)!=size or hashlib.sha384(data).hexdigest()!=digest:
            raise RuntimeError(f"Downloaded model checksum mismatch: {target.name}")
        temporary = target.with_suffix(target.suffix+".part")
        temporary.write_bytes(data)
        temporary.replace(target)
        print(f"Downloaded and verified {target}")


if __name__ == "__main__":
    main()
