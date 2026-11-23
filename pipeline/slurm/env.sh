# Carga el entorno usado en el clúster I3US (PyTorch 1.13.1 + CUDA 11.8)
source /etc/profile
ml torchvision/0.14.1-foss-2022a-CUDA-11.8.0 >/dev/null 2>&1
export PYTHONUNBUFFERED=1
