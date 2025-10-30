#!/bin/bash

# ============================================================================
# Leaffliction - Setup Script
# ============================================================================
# Este script configura el entorno virtual y todas las dependencias necesarias
# para ejecutar el proyecto de clasificación de enfermedades en hojas.
# ============================================================================

set -e  # Salir si hay algún error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir mensajes
print_message() {
    echo -e "${GREEN}[SETUP]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Banner
echo -e "${GREEN}"
echo "============================================================"
echo "   🌿 LEAFFLICTION - PLANT DISEASE CLASSIFICATION 🌿"
echo "============================================================"
echo -e "${NC}"

# 1. Verificar Python
print_message "Verificando instalación de Python..."
if ! command -v python3 &> /dev/null; then
    print_error "Python3 no está instalado. Por favor, instala Python 3.8 o superior."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
print_info "Python $PYTHON_VERSION detectado ✓"

# 2. Verificar pip
print_message "Verificando pip..."
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 no está instalado. Instalando..."
    python3 -m ensurepip --default-pip
fi
print_info "pip está disponible ✓"

# 3. Crear entorno virtual
VENV_DIR="venv"
print_message "Creando entorno virtual en '$VENV_DIR'..."
if [ -d "$VENV_DIR" ]; then
    print_warning "El directorio '$VENV_DIR' ya existe."
    read -p "¿Deseas eliminarlo y crear uno nuevo? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        print_info "Directorio eliminado."
    else
        print_info "Usando el entorno virtual existente."
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    print_info "Entorno virtual creado ✓"
fi

# 4. Activar entorno virtual
print_message "Activando entorno virtual..."
source "$VENV_DIR/bin/activate"

# 5. Actualizar pip
print_message "Actualizando pip..."
pip install --upgrade pip --quiet

# 6. Crear requirements.txt si no existe
REQUIREMENTS_FILE="requirements.txt"
print_message "Verificando archivo de dependencias..."

if [ ! -f "$REQUIREMENTS_FILE" ]; then
    print_warning "No se encontró $REQUIREMENTS_FILE. Creando uno basado en el código..."
    cat > "$REQUIREMENTS_FILE" << EOF
# Core dependencies
numpy>=1.21.0
opencv-python>=4.5.0
matplotlib>=3.3.0
Pillow>=8.0.0

# Machine Learning
torch>=2.0.0
torchvision>=0.15.0
scikit-learn>=0.24.0
tqdm>=4.60.0

# Optional but recommended
flake8>=3.9.0
EOF
    print_info "Archivo $REQUIREMENTS_FILE creado ✓"
fi

# 7. Instalar dependencias
print_message "Instalando dependencias (esto puede tardar varios minutos)..."
echo ""
pip install -r "$REQUIREMENTS_FILE"
echo ""
print_info "Dependencias instaladas ✓"

# 8. Verificar instalación de CUDA (opcional para GPU)
print_message "Verificando soporte GPU..."
python3 << EOF
import torch
if torch.cuda.is_available():
    print(f"✓ GPU disponible: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA version: {torch.version.cuda}")
else:
    print("⚠ No se detectó GPU. El entrenamiento se hará en CPU (será más lento).")
EOF

# 9. Verificar importaciones críticas
print_message "Verificando que todas las librerías se importan correctamente..."
python3 << EOF
import sys
errors = []

try:
    import cv2
    print("✓ opencv-python (cv2)")
except ImportError as e:
    errors.append(f"✗ opencv-python: {e}")

try:
    import numpy
    print("✓ numpy")
except ImportError as e:
    errors.append(f"✗ numpy: {e}")

try:
    import matplotlib
    print("✓ matplotlib")
except ImportError as e:
    errors.append(f"✗ matplotlib: {e}")

try:
    import PIL
    print("✓ Pillow (PIL)")
except ImportError as e:
    errors.append(f"✗ Pillow: {e}")

try:
    import torch
    print("✓ torch (PyTorch)")
except ImportError as e:
    errors.append(f"✗ torch: {e}")

try:
    import torchvision
    print("✓ torchvision")
except ImportError as e:
    errors.append(f"✗ torchvision: {e}")

try:
    import sklearn
    print("✓ scikit-learn")
except ImportError as e:
    errors.append(f"✗ scikit-learn: {e}")

try:
    import tqdm
    print("✓ tqdm")
except ImportError as e:
    errors.append(f"✗ tqdm: {e}")

if errors:
    print("\n❌ Errores encontrados:")
    for error in errors:
        print(error)
    sys.exit(1)
else:
    print("\n✅ Todas las librerías están correctamente instaladas")
EOF

if [ $? -ne 0 ]; then
    print_error "Algunas librerías no se pudieron importar."
    print_warning "Intenta instalar manualmente las que fallaron."
    exit 1
fi

# 10. Crear estructura de directorios
print_message "Creando estructura de directorios..."
mkdir -p model_output
mkdir -p logs
print_info "Directorios creados ✓"

# 11. Verificar flake8 (norminette Python)
print_message "Verificando estilo de código (flake8)..."
if command -v flake8 &> /dev/null; then
    print_info "Ejecutando flake8..."
    flake8 *.py --max-line-length=120 --ignore=E501,W503 || true
else
    print_warning "flake8 no está instalado. Instálalo con: pip install flake8"
fi

# 12. Resumen final
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}   ✅ SETUP COMPLETADO EXITOSAMENTE${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
print_info "Para activar el entorno virtual en el futuro:"
echo -e "   ${YELLOW}source $VENV_DIR/bin/activate${NC}"
echo ""
print_info "Para desactivar el entorno virtual:"
echo -e "   ${YELLOW}deactivate${NC}"
echo ""
print_info "Scripts disponibles:"
echo "   • Distribution.py  - Análisis del dataset"
echo "   • Augmentation.py  - Aumento de datos"
echo "   • Balance.py       - Balanceo automático del dataset"
echo "   • Transformation.py - Transformaciones de imágenes"
echo "   • train.py         - Entrenamiento del modelo"
echo "   • predict.py       - Predicción de enfermedades"
echo ""
print_info "Ejemplo de uso:"
echo -e "   ${YELLOW}python3 Distribution.py Apple${NC}"
echo -e "   ${YELLOW}python3 train.py ./Apple/${NC}"
echo -e "   ${YELLOW}python3 predict.py ./Apple/apple_healthy/image.jpg${NC}"
echo ""
print_warning "Recuerda: Necesitas descargar el dataset antes de entrenar."
print_warning "El dataset NO debe estar en el repositorio Git."
echo ""
echo -e "${GREEN}🌿 ¡Listo para empezar! 🌿${NC}"
echo ""