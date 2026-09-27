import os
import sys
import zipfile

SKIP = [
    ".DS_Store",
    "__pycache__",
    ".ipynb_checkpoints",
    ".pyc",
    ".pyo",
    ".log",
    ".swp",
    ".tmp",
    ".bak",
]


def skip(name):
    return name.startswith("._") or any(s in name for s in SKIP)


root = os.path.dirname(os.path.abspath(__file__))
folders = ["models", "mytorch"]
zip_path = os.path.join(root, "handin.zip")

for folder in folders:
    if not os.path.isdir(os.path.join(root, folder)):
        print(f"❌ Missing folder: {folder}")
        sys.exit(1)

print("📦 Creating handin.zip...")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for folder in folders:
        for walk_root, dirs, files in os.walk(os.path.join(root, folder)):
            dirs[:] = [d for d in dirs if not skip(d)]
            for f in files:
                path = os.path.join(walk_root, f)
                rel = os.path.relpath(path, root)
                if skip(f):
                    print(f"  ⏭ {rel}")
                    continue
                zf.write(path, rel)
                print(f"  ✅ {rel}")

print("🎉 Done! Upload handin.zip to Gradescope.")
