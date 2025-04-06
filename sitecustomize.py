# các khởi tạo ban đầu dự án
# cấu hình cho vị trí đứng là ROOT dự án
# khi chạy setup.bat, file sitecustomize.py sẽ được copy vào venv/Lib/site-packages và tự động run cũng như caching lại

import sys
import os


# thêm root vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
while current_dir and not os.path.exists(os.path.join(current_dir, "venv")):
    current_dir = os.path.dirname(current_dir)

ROOT = current_dir
if ROOT not in sys.path:
    sys.path.append(ROOT)

print(f"ROOT: {ROOT}")