#!/usr/bin/env python3
"""
predict.py - Predice la enfermedad de una hoja usando el modelo entrenado
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Configuración
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = 'model_output'
IMG_SIZE = 224


class PlantCNN(nn.Module):
    """
    Misma arquitectura que en train.py
    """
    def __init__(self, num_classes):
        super(PlantCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.flatten_size = 256 * 14 * 14
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        return x


def load_model(model_path, class_names_path):
    """
    Carga el modelo entrenado y los nombres de clases.
    
    Returns:
        model: Modelo cargado
        class_names: Lista de nombres de clases
    """
    # Cargar nombres de clases
    if not os.path.exists(class_names_path):
        print(f"Error: No se encuentra {class_names_path}")
        sys.exit(1)
    
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    
    print(f"✓ Clases cargadas: {len(class_names)}")
    
    # Cargar modelo
    if not os.path.exists(model_path):
        print(f"Error: No se encuentra {model_path}")
        sys.exit(1)
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    num_classes = checkpoint['num_classes']
    
    # Crear modelo y cargar pesos
    model = PlantCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    print(f"✓ Modelo cargado desde {model_path}")
    
    return model, class_names


def get_transform():
    """
    Transforma la imagen para predicción (mismas que validation).
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def predict_image(model, image_path, class_names):
    """
    Hace predicción sobre una imagen.
    
    Returns:
        predicted_class: Nombre de la clase predicha
        confidence: Confianza de la predicción (0-100)
        probabilities: Probabilidades de todas las clases
    """
    # Cargar imagen
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error cargando imagen: {e}")
        sys.exit(1)
    
    # Guardar imagen original para visualización
    original_image = image.copy()
    
    # Transformar imagen
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0)  # Añadir dimensión de batch
    image_tensor = image_tensor.to(DEVICE)
    
    # Predicción
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    # Convertir a valores de Python
    predicted_idx = predicted_idx.item()
    confidence = confidence.item() * 100
    probabilities = probabilities.cpu().numpy()[0]
    
    predicted_class = class_names[predicted_idx]
    
    return predicted_class, confidence, probabilities, original_image


def denormalize_image(tensor):
    """
    Desnormaliza un tensor para visualización.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    return tensor


def visualize_prediction(original_image, predicted_class, confidence, probabilities, class_names, save_path=None):
    """
    Visualiza la predicción con la imagen original, transformada y resultados.
    """
    # Crear figura
    fig = plt.figure(figsize=(16, 6))
    
    # Gridspec para layout personalizado
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.2], hspace=0.3, wspace=0.3)
    
    # 1. Imagen original
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(original_image)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 2. Imagen transformada (redimensionada)
    ax2 = fig.add_subplot(gs[:, 1])
    transformed = original_image.resize((IMG_SIZE, IMG_SIZE))
    ax2.imshow(transformed)
    ax2.set_title('Transformed Image\n(Resized to 224x224)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # 3. Resultado de la predicción (texto grande)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    result_text = "=== DL classification ===\n\n"
    result_text += f"Class predicted:\n{predicted_class}\n\n"
    result_text += f"Confidence: {confidence:.1f}%"
    
    # Color según confianza
    if confidence >= 90:
        color = 'green'
    elif confidence >= 70:
        color = 'orange'
    else:
        color = 'red'
    
    ax3.text(0.5, 0.5, result_text, 
            fontsize=16, 
            ha='center', 
            va='center',
            bbox=dict(boxstyle='round,pad=1', facecolor='wheat', alpha=0.8),
            family='monospace')
    
    # 4. Gráfico de barras con todas las probabilidades
    ax4 = fig.add_subplot(gs[1, 2])
    
    # Ordenar por probabilidad
    sorted_indices = np.argsort(probabilities)[::-1]
    top_n = min(5, len(class_names))  # Top 5 o menos
    
    top_indices = sorted_indices[:top_n]
    top_probs = probabilities[top_indices] * 100
    top_classes = [class_names[i] for i in top_indices]
    
    # Colores: verde para la predicción, azul para el resto
    colors = ['green' if i == sorted_indices[0] else 'skyblue' for i in range(top_n)]
    
    bars = ax4.barh(range(top_n), top_probs, color=colors, alpha=0.7)
    ax4.set_yticks(range(top_n))
    ax4.set_yticklabels(top_classes, fontsize=10)
    ax4.set_xlabel('Probability (%)', fontsize=11)
    ax4.set_title(f'Top {top_n} Predictions', fontsize=12, fontweight='bold')
    ax4.set_xlim(0, 100)
    ax4.grid(axis='x', alpha=0.3)
    
    # Añadir valores en las barras
    for i, (bar, prob) in enumerate(zip(bars, top_probs)):
        ax4.text(prob + 2, i, f'{prob:.1f}%', 
                va='center', fontsize=9, fontweight='bold')
    
    plt.suptitle('Plant Disease Classification', fontsize=18, fontweight='bold', y=0.98)
    
    # Guardar o mostrar
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualización guardada en {save_path}")
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Función principal.
    """
    # Verificar argumentos
    if len(sys.argv) != 2:
        print("Uso: python3 predict.py <imagen>")
        print("Ejemplo: python3 predict.py ./test_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Verificar que existe la imagen
    if not os.path.exists(image_path):
        print(f"Error: La imagen {image_path} no existe")
        sys.exit(1)
    
    print("=" * 60)
    print("🔮 PREDICCIÓN DE ENFERMEDAD EN HOJAS")
    print("=" * 60)
    
    # Cargar modelo
    print("\n📦 Cargando modelo...")
    model_path = os.path.join(MODEL_DIR, 'plant_disease_model.pth')
    class_names_path = os.path.join(MODEL_DIR, 'class_names.json')
    
    model, class_names = load_model(model_path, class_names_path)
    
    # Hacer predicción
    print(f"\n🔍 Analizando imagen: {image_path}")
    predicted_class, confidence, probabilities, original_image = predict_image(
        model, image_path, class_names
    )
    
    # Mostrar resultado en consola
    print("\n" + "=" * 60)
    print("📊 RESULTADO")
    print("=" * 60)
    print(f"\n✓ Clase predicha:  {predicted_class}")
    print(f"✓ Confianza:       {confidence:.2f}%")
    
    print(f"\nTop 3 predicciones:")
    sorted_indices = np.argsort(probabilities)[::-1]
    for i in range(min(3, len(class_names))):
        idx = sorted_indices[i]
        print(f"  {i+1}. {class_names[idx]:<30} {probabilities[idx]*100:>6.2f}%")
    
    # Visualizar
    print("\n📊 Generando visualización...")
    save_path = image_path.replace(os.path.splitext(image_path)[1], '_prediction.png')
    visualize_prediction(
        original_image, 
        predicted_class, 
        confidence, 
        probabilities, 
        class_names,
        save_path
    )
    
    print("\n" + "=" * 60)
    print("✅ PREDICCIÓN COMPLETADA")
    print("=" * 60)


if __name__ == "__main__":
    main()