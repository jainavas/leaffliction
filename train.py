#!/usr/bin/env python3
"""
train.py - Entrena un modelo CNN para clasificación de enfermedades en hojas
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

# Configuración del dispositivo
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Usando dispositivo: {DEVICE}")

if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memoria disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Hiperparámetros
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
IMG_SIZE = 224
TRAIN_SPLIT = 0.8  # 80% training, 20% validation

class PlantDiseaseDataset(Dataset):
    """
    Dataset personalizado para cargar imágenes de enfermedades de plantas.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Cargar imagen
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error cargando {img_path}: {e}")
            # Retornar imagen negra en caso de error
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        
        label = self.labels[idx]
        
        # Aplicar transformaciones
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_dataset(data_dir):
    """
    Carga todas las imágenes y sus etiquetas desde un directorio.
    
    Estructura esperada:
        data_dir/
            clase1/
                img1.jpg
                img2.jpg
            clase2/
                img1.jpg
    
    Returns:
        image_paths: Lista de paths a imágenes
        labels: Lista de índices de clase
        class_names: Lista de nombres de clases
    """
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} no existe")
        sys.exit(1)
    
    image_paths = []
    labels = []
    
    # Obtener nombres de clases (subdirectorios)
    class_names = sorted([d for d in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, d))])
    
    if not class_names:
        print(f"Error: No se encontraron subdirectorios (clases) en {data_dir}")
        sys.exit(1)
    
    print(f"\n📊 Clases encontradas: {len(class_names)}")
    print("-" * 60)
    
    # Extensiones válidas
    extensiones = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG')
    
    # Cargar imágenes de cada clase
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        
        # Listar imágenes
        images = [f for f in os.listdir(class_dir) if f.endswith(extensiones)]
        
        print(f"  [{class_idx}] {class_name:<30} {len(images):>6} imágenes")
        
        # Agregar a las listas
        for img_name in images:
            img_path = os.path.join(class_dir, img_name)
            image_paths.append(img_path)
            labels.append(class_idx)
    
    print("-" * 60)
    print(f"✓ Total: {len(image_paths)} imágenes cargadas\n")
    
    return image_paths, labels, class_names

def get_transforms(train=True):
    """
    Define las transformaciones para training y validation.
    
    Args:
        train: Si True, aplica augmentation. Si False, solo normalización.
    """
    if train:
        # Augmentation para training
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                 saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            # Normalización con valores estándar de ImageNet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Solo resize y normalización para validation
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
class PlantCNN(nn.Module):
    """
    Red neuronal convolucional personalizada para clasificación de enfermedades.
    """
    def __init__(self, num_classes):
        super(PlantCNN, self).__init__()
        
        # Bloque convolucional 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Output: 112x112x32
        
        # Bloque convolucional 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Output: 56x56x64
        
        # Bloque convolucional 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Output: 28x28x128
        
        # Bloque convolucional 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Output: 14x14x256
        
        # Calcular tamaño después de las convoluciones
        # IMG_SIZE=224 → después de 4 maxpool(2): 224/(2^4) = 14
        self.flatten_size = 256 * 14 * 14
        
        # Capas fully connected
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
        """
        Forward pass de la red.
        
        Args:
            x: Batch de imágenes [batch_size, 3, 224, 224]
        
        Returns:
            Logits [batch_size, num_classes]
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        return x
    
def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Entrena el modelo por una época.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Barra de progreso
    pbar = tqdm(dataloader, desc='Training')
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Métricas
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Actualizar barra de progreso
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
    Valida el modelo.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Validation')
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Actualizar barra de progreso
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc

def plot_metrics(train_losses, train_accs, val_losses, val_accs, save_path='training_metrics.png'):
    """
    Grafica las métricas de entrenamiento y validación.
    """
    epochs_range = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(epochs_range, train_losses, 'b-o', label='Training Loss', linewidth=2)
    ax1.plot(epochs_range, val_losses, 'r-o', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs_range, train_accs, 'b-o', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs_range, val_accs, 'r-o', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=90, color='g', linestyle='--', label='90% threshold', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Gráfica guardada en {save_path}")
    plt.close()
    
def save_model(model, class_names, train_accs, val_accs, train_losses, val_losses, save_dir='model_output'):
    """
    Guarda el modelo entrenado y metadata asociada.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Guardar modelo
    model_path = os.path.join(save_dir, 'plant_disease_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': len(class_names),
        'img_size': IMG_SIZE,
    }, model_path)
    print(f"✓ Modelo guardado en {model_path}")
    
    # Guardar class names
    class_names_path = os.path.join(save_dir, 'class_names.json')
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f, indent=4)
    print(f"✓ Nombres de clases guardados en {class_names_path}")
    
    # Guardar métricas
    metrics_path = os.path.join(save_dir, 'training_metrics.json')
    metrics = {
        'train_losses': train_losses,
        'train_accuracies': train_accs,
        'val_losses': val_losses,
        'val_accuracies': val_accs,
        'final_train_acc': train_accs[-1],
        'final_val_acc': val_accs[-1],
        'best_val_acc': max(val_accs),
        'epochs': len(train_losses),
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'img_size': IMG_SIZE
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"✓ Métricas guardadas en {metrics_path}")
    
    # Guardar gráfica
    plot_path = os.path.join(save_dir, 'training_metrics.png')
    plot_metrics(train_losses, train_accs, val_losses, val_accs, plot_path)

def main():
    """
    Función principal de entrenamiento.
    """
    # Verificar argumentos
    if len(sys.argv) != 2:
        print("Uso: python3 train.py <directorio_dataset>")
        print("Ejemplo: python3 train.py ./Apple/")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    
    print("=" * 60)
    print("🌿 ENTRENAMIENTO DE MODELO - CLASIFICACIÓN DE ENFERMEDADES")
    print("=" * 60)
    
    # 1. Cargar dataset
    print("\n📂 Cargando dataset...")
    image_paths, labels, class_names = load_dataset(data_dir)
    
    # Verificar que hay suficientes datos
    if len(image_paths) < 100:
        print(f"⚠️  Advertencia: Solo hay {len(image_paths)} imágenes. Se recomienda >1000 para buenos resultados.")
    
    # 2. Split train/validation
    print(f"\n✂️  Dividiendo dataset ({TRAIN_SPLIT*100:.0f}% train, {(1-TRAIN_SPLIT)*100:.0f}% val)...")
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, 
        test_size=(1-TRAIN_SPLIT), 
        stratify=labels,  # Mantener proporción de clases
        random_state=42
    )
    
    print(f"  Training:   {len(X_train)} imágenes")
    print(f"  Validation: {len(X_val)} imágenes")
    
    # Verificar validation set
    if len(X_val) < 100:
        print(f"⚠️  Advertencia: Validation set tiene solo {len(X_val)} imágenes. Se requiere mínimo 100.")
    
    # 3. Crear datasets y dataloaders
    print("\n🔄 Creando dataloaders...")
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)
    
    train_dataset = PlantDiseaseDataset(X_train, y_train, transform=train_transform)
    val_dataset = PlantDiseaseDataset(X_val, y_val, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,  # Carga paralela (ajusta según tu CPU)
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 4. Crear modelo
    print(f"\n🧠 Creando modelo CNN con {len(class_names)} clases...")
    model = PlantCNN(num_classes=len(class_names))
    model = model.to(DEVICE)
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parámetros totales: {total_params:,}")
    print(f"  Parámetros entrenables: {trainable_params:,}")
    
    # 5. Loss y optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler (reduce LR cuando val_loss deja de mejorar)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 6. Entrenamiento
    print("\n" + "=" * 60)
    print("🚀 INICIANDO ENTRENAMIENTO")
    print("=" * 60)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\n📅 Epoch {epoch+1}/{EPOCHS}")
        print("-" * 60)
        
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Guardar métricas
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Imprimir resumen
        print(f"\n📊 Resumen Epoch {epoch+1}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  ⭐ Nuevo mejor accuracy: {best_val_acc:.2f}%")
            # Podrías guardar checkpoint aquí
        
        # Early stopping si alcanzamos 95%
        if val_acc >= 95.0:
            print(f"\n🎉 Alcanzado {val_acc:.2f}% accuracy en validación!")
            print("   Deteniendo entrenamiento anticipadamente.")
            break
    
    # 7. Resultados finales
    print("\n" + "=" * 60)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"\n📈 Resultados finales:")
    print(f"  Mejor Val Accuracy:  {best_val_acc:.2f}%")
    print(f"  Final Train Acc:     {train_accs[-1]:.2f}%")
    print(f"  Final Val Acc:       {val_accs[-1]:.2f}%")
    
    # Verificar requisito del proyecto
    if val_accs[-1] >= 90.0:
        print(f"\n✅ CUMPLE REQUISITO: Accuracy > 90% ({val_accs[-1]:.2f}%)")
    else:
        print(f"\n⚠️  NO CUMPLE REQUISITO: Accuracy < 90% ({val_accs[-1]:.2f}%)")
        print("   Considera entrenar más épocas o ajustar hiperparámetros")
    
    # 8. Guardar modelo
    print("\n💾 Guardando modelo y métricas...")
    save_model(model, class_names, train_accs, val_accs, train_losses, val_losses)
    
    print("\n" + "=" * 60)
    print("🎉 ¡PROCESO COMPLETADO!")
    print("=" * 60)
    print("\nArchivos generados:")
    print("  📁 model_output/")
    print("     ├── plant_disease_model.pth")
    print("     ├── class_names.json")
    print("     ├── training_metrics.json")
    print("     └── training_metrics.png")
    print("\nPuedes usar predict.py para hacer predicciones con el modelo entrenado.")


if __name__ == "__main__":
    main()