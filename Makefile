.PHONY: help setup clean test lint train predict distribution augmentation

# Configuración
VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

# Color output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
NC = \033[0m # No Color

help: ## Muestra esta ayuda
	@echo "$(GREEN)Leaffliction - Comandos disponibles:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

setup: ## Configura el entorno virtual e instala dependencias
	@echo "$(GREEN)[SETUP]$(NC) Ejecutando setup.sh..."
	@chmod +x setup.sh
	@./setup.sh

clean: ## Limpia archivos temporales y caché
	@echo "$(YELLOW)[CLEAN]$(NC) Limpiando archivos temporales..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name ".DS_Store" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓$(NC) Limpieza completada"

clean-all: clean ## Limpia todo incluyendo el venv
	@echo "$(RED)[CLEAN-ALL]$(NC) Eliminando entorno virtual..."
	@rm -rf $(VENV)
	@rm -rf model_output
	@echo "$(GREEN)✓$(NC) Limpieza completa"

lint: ## Ejecuta flake8 para verificar estilo de código
	@echo "$(YELLOW)[LINT]$(NC) Verificando código con flake8..."
	@$(VENV)/bin/flake8 *.py --max-line-length=120 --ignore=E501,W503 || true

test: lint ## Ejecuta tests básicos
	@echo "$(YELLOW)[TEST]$(NC) Verificando imports..."
	@$(PYTHON) -c "import cv2, numpy, torch, torchvision, sklearn, matplotlib; print('$(GREEN)✓$(NC) Todas las librerías OK')"

# Comandos del proyecto
distribution: ## Ejecuta Distribution.py (usa: make distribution DIR=./Apple)
	@echo "$(GREEN)[DISTRIBUTION]$(NC) Analizando dataset..."
	@$(PYTHON) Distribution.py $(DIR)

augmentation: ## Ejecuta Augmentation.py (usa: make augmentation IMG=./image.jpg)
	@echo "$(GREEN)[AUGMENTATION]$(NC) Aumentando imagen..."
	@$(PYTHON) Augmentation.py $(IMG) 1

balance: ## Ejecuta Balance.py (usa: make balance FILTER=Apple)
	@echo "$(GREEN)[BALANCE]$(NC) Balanceando dataset..."
	@$(PYTHON) Balance.py $(FILTER)

transformation: ## Ejecuta Transformation.py (usa: make transformation IMG=./image.jpg)
	@echo "$(GREEN)[TRANSFORMATION]$(NC) Transformando imagen..."
	@$(PYTHON) Transformation.py $(IMG)

train: ## Entrena el modelo (usa: make train DATA=./Apple/)
	@echo "$(GREEN)[TRAIN]$(NC) Entrenando modelo..."
	@$(PYTHON) train.py $(DATA)

predict: ## Predice enfermedad (usa: make predict IMG=./test.jpg)
	@echo "$(GREEN)[PREDICT]$(NC) Prediciendo enfermedad..."
	@$(PYTHON) predict.py $(IMG)

# Comandos útiles
freeze: ## Genera requirements.txt actualizado
	@echo "$(YELLOW)[FREEZE]$(NC) Generando requirements.txt..."
	@$(PIP) freeze > requirements.txt
	@echo "$(GREEN)✓$(NC) requirements.txt actualizado"

upgrade: ## Actualiza todas las dependencias
	@echo "$(YELLOW)[UPGRADE]$(NC) Actualizando dependencias..."
	@$(PIP) install --upgrade -r requirements.txt

venv-activate: ## Muestra comando para activar venv
	@echo "$(YELLOW)Para activar el entorno virtual ejecuta:$(NC)"
	@echo "  source $(VENV)/bin/activate"

check-gpu: ## Verifica si GPU está disponible
	@echo "$(YELLOW)[GPU CHECK]$(NC) Verificando GPU..."
	@$(PYTHON) -c "import torch; print('GPU disponible:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

install: setup ## Alias para setup

.DEFAULT_GOAL := help