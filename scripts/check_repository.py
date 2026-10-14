"""Reject tracked local research assets before publishing repository changes."""

import subprocess
from pathlib import Path, PurePosixPath

LOCAL_DIRECTORIES = {"data", "models", "outputs", "logs", ".ssh", ".venv", "venv", "env"}
BLOCKED_SUFFIXES = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".onnx",
    ".pt",
    ".pth",
    ".ckpt",
    ".h5",
    ".hdf5",
    ".safetensors",
    ".npy",
    ".npz",
    ".pem",
    ".key",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".7z",
}
DIRECTORY_DOCUMENTATION = {"data/README.md", "models/README.md", "outputs/README.md"}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True)
    paths = [
        path.decode("utf-8", errors="surrogateescape")
        for path in result.stdout.split(b"\0")
        if path
    ]
    blocked = []
    for name in paths:
        path = PurePosixPath(name)
        if name in DIRECTORY_DOCUMENTATION:
            continue
        local_asset = any(part in LOCAL_DIRECTORIES for part in path.parts)
        credential = path.name == ".env" or (
            path.name.startswith(".env.") and path.name != ".env.example"
        )
        if local_asset or credential or path.suffix.lower() in BLOCKED_SUFFIXES:
            blocked.append(name)

    if blocked:
        print("Remove these local assets from the Git index before publishing:")
        for name in blocked:
            print(f"  {name}")
        return 1
    print(f"Repository file check passed ({len(paths)} tracked files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
