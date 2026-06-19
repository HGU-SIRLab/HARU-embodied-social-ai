#!/usr/bin/env bash
# =============================================================================
# HARU Social VLA - Jetson AGX Orin (JetPack 6.x / CUDA 12.x) 환경 설치 스크립트
# Python 3.10 | ARM64 | haru_social_vla/haru_vla_env 가상환경
# =============================================================================

set -e

# --------------------------------------------------------------------------- #
# 0. 경로 설정 (scripts/ 한 단계 위가 PROJECT_DIR)
# --------------------------------------------------------------------------- #
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPTS_DIR}")"
VENV_DIR="${PROJECT_DIR}/haru_vla_env"
WHEEL_REPO="/home/herobot/robot_brain_workspace"
TORCH_WHEEL="${WHEEL_REPO}/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"

echo "============================================================"
echo " HARU Social VLA - Environment Setup"
echo " Jetson AGX Orin | JetPack 6.x | CUDA 12.x | Python 3.10"
echo " Project : ${PROJECT_DIR}"
echo "============================================================"

# --------------------------------------------------------------------------- #
# 1. 가상환경 생성 (없으면 자동 생성)
# --------------------------------------------------------------------------- #
if [ ! -d "${VENV_DIR}" ]; then
    echo "[INFO] 가상환경이 없습니다. 새로 생성합니다: ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
echo "[OK] 가상환경 활성화: ${VENV_DIR}"

# --------------------------------------------------------------------------- #
# 2. pip / setuptools / wheel 업그레이드
# --------------------------------------------------------------------------- #
pip install --upgrade pip wheel setuptools --quiet
echo "[OK] pip 업그레이드 완료"

# --------------------------------------------------------------------------- #
# 3. PyTorch (Jetson 전용 wheel) — --no-deps 로 PyPI torch 교체 방지
#    ⚠️  일반 pip install torch 는 CUDA 비활성 torch 설치 → 절대 사용 금지
# --------------------------------------------------------------------------- #
echo ""
echo "[STEP 1/4] PyTorch (Jetson ARM64 wheel) 설치 중..."

if python -c "import torch; assert 'nv24' in torch.__version__" &>/dev/null; then
    echo "[SKIP] Jetson PyTorch 이미 설치됨: $(python -c 'import torch; print(torch.__version__)')"
elif [ -f "${TORCH_WHEEL}" ]; then
    pip install --no-deps "${TORCH_WHEEL}"
    pip install "sympy==1.13.1" "filelock" "networkx" "jinja2" "fsspec" "typing-extensions>=4.8.0" --quiet
    echo "[OK] 로컬 wheel 에서 PyTorch 설치 완료"
else
    NVIDIA_WHEEL_URL="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
    echo "[INFO] 로컬 wheel 없음 → NVIDIA 서버에서 다운로드 (~1.5 GB)"
    pip install --no-deps "${NVIDIA_WHEEL_URL}"
    pip install "sympy==1.13.1" "filelock" "networkx" "jinja2" "fsspec" "typing-extensions>=4.8.0" --quiet
fi

# --------------------------------------------------------------------------- #
# 4. torchvision — Jetson 사전 빌드 egg 를 site-packages 에 직접 압축 해제
#    ⚠️  PyPI torchvision 설치 금지: torch 최신 버전(+CUDA 13.x) 을 끌어옴
# --------------------------------------------------------------------------- #
echo ""
echo "[STEP 2/4] torchvision (Jetson ARM64 egg) 설치 중..."

TORCHVISION_EGG="${WHEEL_REPO}/torchvision_src/dist/torchvision-0.20.0-py3.10-linux-aarch64.egg"
SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")

if python -c "import torchvision" &>/dev/null; then
    echo "[SKIP] torchvision 이미 설치됨: $(python -c 'import torchvision; print(torchvision.__version__)')"
elif [ -f "${TORCHVISION_EGG}" ]; then
    unzip -q -o "${TORCHVISION_EGG}" -d "${SITE_PKG}"
    echo "[OK] torchvision egg 설치 완료"
else
    echo "[WARN] torchvision egg 없음 — timm 일부 기능 제한될 수 있음"
fi

# --------------------------------------------------------------------------- #
# 5. OpenVLA 추론 핵심 패키지
#    - numpy <2.0: Jetson torch 2.5 가 numpy 1.x 기준 컴파일됨
#    - timm 0.9.12 --no-deps: pip 의존성 해석이 PyPI torch 를 끌어오는 것을 방지
#    - tokenizers <0.22: transformers 4.x 호환
# --------------------------------------------------------------------------- #
echo ""
echo "[STEP 3/4] OpenVLA 추론 핵심 패키지 설치 중..."

pip install \
    "transformers>=4.40.0,<5.0.0" \
    "tokenizers>=0.19.0,<0.22" \
    "accelerate>=0.27.0" \
    "safetensors>=0.4.3" \
    "huggingface-hub>=0.23.0" \
    "Pillow>=10.0.0" \
    "numpy>=1.26.0,<2.0"

pip install "timm==0.9.12" --no-deps

# --------------------------------------------------------------------------- #
# 6. 유틸리티 패키지 (opencv <4.11: 4.11+ 은 numpy>=2 요구)
# --------------------------------------------------------------------------- #
echo ""
echo "[STEP 4/4] 유틸리티 패키지 설치 중..."

pip install \
    "opencv-python-headless>=4.8.1,<4.11" \
    "scipy>=1.12.0" \
    "einops>=0.7.0"

# --------------------------------------------------------------------------- #
# 6. 설치 검증
# --------------------------------------------------------------------------- #
echo ""
echo "[STEP 5/5] 설치 검증 중..."
echo "============================================================"

python - <<'EOF'
import sys
results = []

def chk(label, fn):
    try:
        msg = fn()
        results.append(("OK  ", f"{label}: {msg}"))
    except Exception as e:
        results.append(("FAIL", f"{label}: {e}"))

chk("PyTorch",
    lambda: __import__('torch').__version__ + " | CUDA=" + str(__import__('torch').cuda.is_available()))
chk("GPU",
    lambda: __import__('torch').cuda.get_device_name(0) if __import__('torch').cuda.is_available() else "N/A")
chk("transformers",
    lambda: __import__('transformers').__version__)
chk("accelerate",
    lambda: __import__('accelerate').__version__)
chk("timm",
    lambda: __import__('timm').__version__)
chk("Pillow",
    lambda: __import__('PIL').__version__)
chk("numpy",
    lambda: __import__('numpy').__version__)
chk("einops",
    lambda: __import__('einops').__version__)

for status, msg in results:
    print(f"  [{status}] {msg}")

if any(s == "FAIL" for s, _ in results):
    print("\n[ERROR] 일부 패키지 설치 실패. 위 로그를 확인하세요.")
    sys.exit(1)
else:
    print("\n[ALL OK] 모든 패키지 정상 설치 완료.")
EOF

echo "============================================================"
echo " 설치 완료"
echo " 가상환경 활성화: source ${VENV_DIR}/bin/activate"
echo "============================================================"
