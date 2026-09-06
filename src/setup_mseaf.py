# !git clone https://github.com/K1etTuan/DACS.git
# %cd /content/DACS
# !git pull
import os
import shutil
import ultralytics


# Lấy đường dẫn ultralytics
base = os.path.dirname(ultralytics.__file__)

print("Ultralytics version:", ultralytics.__version__)
print("Ultralytics path:", base)


# 1. Copy MSEAF.py vào ultralytics/nn/modules
src = os.path.join(os.path.dirname(__file__), "MSEAF.py")
dst = os.path.join(base, "nn", "modules", "MSEAF.py")

shutil.copy(src, dst)
print("Đã copy MSEAF.py")


# 2. Thêm MSEAF vào modules/__init__.py
path = os.path.join(base, "nn", "modules", "__init__.py")

with open(path, "r") as f:
    content = f.read()

line = "from .MSEAF import MSEAF"

if line not in content:
    with open(path, "a") as f:
        f.write("\n" + line + "\n")

print("Đã thêm MSEAF vào __init__.py")


# 3. Thêm MSEAF vào tasks.py
path = os.path.join(base, "nn", "tasks.py")

with open(path, "r") as f:
    content = f.read()

line = "from ultralytics.nn.modules import MSEAF"

if line not in content:
    with open(path, "a") as f:
        f.write("\n" + line + "\n")

print("Đã thêm MSEAF vào tasks.py")

print("Setup MSEAF xong!")

# !pip install -q ultralytics
# !python /content/DACS/src/setup_mseaf.py