"""
scripts/deploy_kaggle.py
------------------------
Đóng gói (1) source code sạch từ Git HEAD, (2) file weights đã train, (3) các
gói pip mà Kaggle Notebook không có sẵn — thành 1 Kaggle Dataset (private),
dùng làm input offline cho Notebook chạy ở chế độ "Internet: Off" của Code
Competition (xem notebooks/kaggle_submission.ipynb).

Chạy từ MÁY LOCAL (không phải trên Kaggle) — cần đã `pip install kaggle` và
đăng nhập Kaggle CLI trước đó (`kaggle config view` để kiểm tra), bằng 1 trong
2 cách: file `kaggle.json` (username+key) đặt ở vị trí mặc định `~/.kaggle/`,
hoặc OAuth (`~/.kaggle/credentials.json`, sinh ra khi chạy `kaggle` lần đầu và
đăng nhập qua trình duyệt). Script này KHÔNG tự set `KAGGLE_CONFIG_DIR` — luôn
dùng đúng credential mặc định mà `kaggle` CLI của bạn đang xác thực được.

Cách dùng:
    # Lần đầu (tạo dataset mới):
    python scripts/deploy_kaggle.py --weights D:\\downloads\\best.pt --slug wheat-yolov8-offline-bundle --new

    # Các lần sau (cập nhật version, dùng chung 1 dataset):
    python scripts/deploy_kaggle.py --weights D:\\downloads\\best.pt --slug wheat-yolov8-offline-bundle -m "Update conf threshold"

Lưu ý: `--weights` phải là file best.pt đã TẢI VỀ MÁY từ tab Output của Kaggle
training notebook (vì pipeline này train trên Kaggle GPU, không train ở local —
xem README/SYSTEM_ARCHITECTURE.md).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# Đường dẫn repo (và nhiều thông báo in ra) chứa ký tự tiếng Việt có dấu. 1 số
# console Windows (đặc biệt Git Bash / cmd cũ) dùng code page không encode được
# hết các ký tự này -> print() có thể crash giữa chừng dù script chạy đúng.
# `errors="replace"` giữ nguyên encoding hiện tại, chỉ thay ký tự không in được
# bằng "?" thay vì raise UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE_DIR = REPO_ROOT / "kaggle_bundle" / "dataset"      # thư mục dàn dựng, build lại mỗi lần chạy

# Các gói pip KHÔNG có sẵn mặc định trên Kaggle Notebook mà src/infer.py cần lúc
# chạy. KHÔNG liệt kê torch/opencv/numpy/pandas/matplotlib/... ở đây — các gói
# đó đã có sẵn trên image Kaggle GPU; tải + cài đè lại chỉ tốn dung lượng dataset
# và có rủi ro ghi đè bản torch có CUDA khớp driver Kaggle bằng bản tải từ PyPI
# không khớp.
#
# `ultralytics==8.4.153` (từ bản >=8.3 trở đi) có thêm dependency bắt buộc
# `ultralytics-platform` (tính năng HUB/telemetry) khi python_version >= 3.11 —
# kéo theo cả 1 chuỗi HTTP client (httpx/httpcore/h11/anyio) + polars. Phát hiện
# thực tế lúc `pip install` trên Kaggle báo lỗi thiếu các gói này (không phải
# đoán trước) — dùng --no-deps từng gói ở đây để KHÔNG kéo theo torch của chúng.
# Nếu lúc test dry-run vẫn gặp `ModuleNotFoundError` cho 1 gói khác, thêm gói đó
# vào đây (hoặc truyền qua --extra-packages) rồi deploy lại.
DEFAULT_PACKAGES = [
    "ultralytics==8.4.153",
    "ultralytics-thop",
    "py-cpuinfo",
    "ultralytics-platform",
    "polars",
    "anyio",
    "h11",
    "httpcore",
    "httpx",
    "cloudpickle",
    "nvidia-ml-py",
    "certifi",
    "charset_normalizer",
    "idna",
    "packaging",
    "python-dateutil",
    "requests",
    "setuptools",
    "six",
    "urllib3",
]

# Runtime mục tiêu của Kaggle Notebook. Các gói ở trên đều là wheel thuần Python
# ("py3-none-any") nên 3 giá trị này thực ra không ảnh hưởng nhiều — nhưng vẫn
# khai báo tường minh để an toàn nếu sau này thêm 1 gói có compiled extension.
# Kiểm tra lại bằng `!python -V` trong 1 Kaggle Notebook (Internet ON) nếu nghi
# ngờ lệch — base image Kaggle có thể đổi Python version theo thời gian.
TARGET_PY_VERSION = "3.11"
TARGET_ABI = "cp311"
TARGET_PLATFORM = "manylinux2014_x86_64"


def run(cmd: list[str], **kwargs) -> None:
    print(f"[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def run_kaggle(cmd: list[str]) -> None:
    """Chạy lệnh `kaggle` CLI, stream output ra console (giữ progress bar upload).

    `kaggle datasets create/version` có nhiều lỗi "logic" (trùng title/slug,
    thiếu quyền...) mà CLI chỉ IN RA dòng chữ rồi vẫn thoát mã 0 — nếu chỉ dựa
    vào exit code như `run()`, script sẽ báo [DONE] dù Kaggle thực ra đã từ
    chối tạo/cập nhật dataset. Nên phải soi thêm chữ "error" trong output."""
    print(f"[CMD] {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    lines: list[str] = []
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    proc.wait()
    output = "".join(lines)

    if proc.returncode != 0:
        raise SystemExit(f"[LỖI] Lệnh kaggle thoát với mã {proc.returncode}.")
    if "error" in output.lower():
        raise SystemExit(
            "\n[LỖI] Kaggle CLI báo lỗi ở output trên (process vẫn thoát mã 0 nên dễ bị tưởng thành công) "
            "-> dataset CHƯA được tạo/cập nhật đúng. Đọc dòng có chữ 'error' ở trên để biết nguyên nhân."
        )


def stage_source(dest: Path, allow_dirty: bool) -> None:
    """git archive HEAD -> giải nén vào `dest`. Chỉ lấy file đã commit & tracked
    (tự động loại .git/, venv/, __pycache__/, data/, *.pt, *.csv... vì các thư
    mục/file này chưa từng được `git add` — xem .gitignore)."""
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    if status.stdout.strip() and not allow_dirty:
        raise SystemExit(
            "[LỖI] Còn thay đổi chưa commit trong repo.\n"
            "  -> `git add` + `git commit` trước khi deploy, để chắc chắn Kaggle chạy đúng\n"
            "     code mới nhất (đây chính là lỗi 'Kaggle chạy nhầm code cũ' đã note trong NOTES.md).\n"
            "  -> Hoặc chạy lại với --allow-dirty nếu chấp nhận đóng gói từ commit HEAD gần nhất,\n"
            "     bỏ qua các thay đổi working-tree chưa commit."
        )

    dest.mkdir(parents=True, exist_ok=True)
    archive_path = dest.parent / "_source.zip"
    run(["git", "archive", "--format=zip", "-o", str(archive_path), "HEAD"], cwd=REPO_ROOT)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(dest)
    archive_path.unlink()
    print(f"[OK] Source (git HEAD) -> {dest}")


def stage_weights(dest: Path, weights_path: Path) -> None:
    if not weights_path.is_file():
        raise SystemExit(
            f"[LỖI] Không tìm thấy file weights: {weights_path}\n"
            "  -> Tải best.pt về máy từ tab Output của Kaggle training notebook trước."
        )
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights_path, dest / "best.pt")
    size_mb = weights_path.stat().st_size / 1e6
    print(f"[OK] Weights: {weights_path} ({size_mb:.1f} MB) -> {dest / 'best.pt'}")


def stage_packages(dest: Path, packages: list[str]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "pip", "download",
        "--no-deps",
        "--only-binary=:all:",
        "--platform", TARGET_PLATFORM,
        "--python-version", TARGET_PY_VERSION,
        "--abi", TARGET_ABI,
        "--implementation", "cp",
        "-d", str(dest),
        *packages,
    ]
    run(cmd)
    print(f"[OK] Packages ({len(packages)}) -> {dest}")


def get_authenticated_username() -> str:
    """Trả về username Kaggle THẬT SỰ đang được `kaggle` CLI xác thực (qua bất
    kỳ cơ chế nào: access token / kaggle.json / OAuth) — KHÔNG đọc trực tiếp
    từ 1 file kaggle.json cụ thể, vì file đó có thể không tồn tại (dùng OAuth)
    hoặc chứa username sai/cũ trong khi CLI thật ra xác thực bằng cơ chế khác
    có độ ưu tiên cao hơn. Dùng sai username ở đây sẽ khiến `dataset-metadata.json`
    ghi owner sai -> Kaggle từ chối tạo dataset (403) dù upload file vẫn chạy được."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    username = api.config_values.get(api.CONFIG_NAME_USER)
    if not username:
        raise SystemExit(
            "[LỖI] Không xác định được username Kaggle đã đăng nhập.\n"
            "  -> Chạy `kaggle config view` để kiểm tra trạng thái đăng nhập,\n"
            "     hoặc `kaggle datasets list --mine` để test thử xác thực."
        )
    return username


def write_dataset_metadata(root: Path, username: str, slug: str, title: str) -> None:
    meta = {
        "title": title,
        "id": f"{username}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (root / "dataset-metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def kaggle_push(root: Path, is_new: bool, message: str, public: bool) -> None:
    if is_new:
        cmd = [sys.executable, "-m", "kaggle", "datasets", "create", "-p", str(root), "-r", "zip"]
        if public:
            cmd.append("-u")
    else:
        cmd = [sys.executable, "-m", "kaggle", "datasets", "version", "-p", str(root), "-r", "zip", "-m", message]

    run_kaggle(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Đóng gói code + weights + wheel packages thành 1 Kaggle Dataset offline."
    )
    parser.add_argument("--weights", required=True, help="Đường dẫn best.pt đã tải về máy.")
    parser.add_argument("--slug", required=True, help='Slug Kaggle Dataset, vd "wheat-yolov8-offline-bundle".')
    parser.add_argument("--title", default=None, help="Tiêu đề dataset (mặc định suy ra từ --slug).")
    parser.add_argument("-m", "--message", default="Update source + weights", help="Version note.")
    parser.add_argument("--new", action="store_true", help="Tạo dataset LẦN ĐẦU (mặc định: version dataset đã có).")
    parser.add_argument("--public", action="store_true", help="Tạo dataset public (mặc định: private).")
    parser.add_argument("--allow-dirty", action="store_true", help="Bỏ qua cảnh báo working-tree chưa commit.")
    parser.add_argument(
        "--extra-packages", nargs="*", default=[],
        help="Thêm gói pip ngoài DEFAULT_PACKAGES (vd 1 dependency thiếu phát hiện lúc test dry-run).",
    )
    args = parser.parse_args()

    username = get_authenticated_username()
    title = args.title or args.slug.replace("-", " ").replace("_", " ").title()

    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)

    stage_source(STAGE_DIR / "source", args.allow_dirty)
    stage_weights(STAGE_DIR / "weights", Path(args.weights))
    stage_packages(STAGE_DIR / "packages", DEFAULT_PACKAGES + list(args.extra_packages))
    write_dataset_metadata(STAGE_DIR, username, args.slug, title)

    print(f"\n[INFO] Bundle sẵn sàng tại: {STAGE_DIR}")
    kaggle_push(STAGE_DIR, args.new, args.message, args.public)

    print(f"\n[DONE] https://www.kaggle.com/datasets/{username}/{args.slug}")
    print("Đợi dataset chuyển trạng thái 'Ready' rồi mới Add vào Notebook để chạy.")


if __name__ == "__main__":
    main()
