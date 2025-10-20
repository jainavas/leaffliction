import os
import sys
import random
from Augmentation import augment  # Corregir typo: Augmantation → Augmentation

def getCategories(filter):
    """
    Busca recursivamente carpetas que contengan 'filter' en su nombre.
    """
    ruta = os.getcwd()
    carpetas = []
    num_archivos = []
    paths_carpetas = []
    max_num_archivos = 0
    carpeta_max = None
    
    extensiones = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG')
    
    for carpeta, subcarpetas, archivos in os.walk(ruta):
        if filter.lower() in os.path.basename(carpeta).lower():
            # Filtrar solo imágenes
            imagenes = [f for f in archivos if f.endswith(extensiones)]
            num_imgs = len(imagenes)
            
            if num_imgs > 0:
                carpetas.append(os.path.basename(carpeta))
                paths_carpetas.append(carpeta)
                num_archivos.append(num_imgs)
                
                if num_imgs > max_num_archivos:
                    max_num_archivos = num_imgs
                    carpeta_max = carpeta
    
    return carpetas, num_archivos, paths_carpetas, carpeta_max, max_num_archivos

def balancear_clase(path_clase, imagenes_actuales, objetivo):
    """
    Balancea una clase específica hasta alcanzar el objetivo.
    """
    a_generar = objetivo - imagenes_actuales
    
    if a_generar <= 0:
        print(f"  ✓ Clase ya balanceada")
        return
    
    print(f"  Generando {a_generar} imágenes...")
    
    extensiones = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG')
    archivos = os.listdir(path_clase)
    imagenes_originales = [f for f in archivos if f.endswith(extensiones)]
    
    if not imagenes_originales:
        print(f"  ✗ Error: no hay imágenes originales")
        return
    
    contador = 0
    
    while contador < a_generar:
        img_nombre = random.choice(imagenes_originales)
        img_path = os.path.join(path_clase, img_nombre)
        
        try:
            augment(img_path, 1)
            contador += 6  # augment genera 6 imágenes
            
            if contador % 60 == 0:
                print(f"    Progreso: {min(contador, a_generar)}/{a_generar}")
        except Exception as e:
            print(f"  ✗ Error procesando {img_nombre}: {e}")
    
    print(f"  ✓ Generadas ~{contador} imágenes")

def main():
    if len(sys.argv) != 2:
        print("Uso: python3 Balance.py <filtro>")
        print("Ejemplo: python3 Balance.py Apple")
        sys.exit(1)
    
    filtro = sys.argv[1]
    
    print(f"Buscando carpetas que contengan '{filtro}'...")
    carpetas, num_archivos, paths, carpeta_max, num_max = getCategories(filtro)
    
    if not carpetas:
        print(f"Error: No se encontraron carpetas con '{filtro}' que contengan imágenes")
        sys.exit(1)
    
    print(f"\nEncontradas {len(carpetas)} carpetas:")
    for nombre, num in zip(carpetas, num_archivos):
        print(f"  - {nombre}: {num} imágenes")
    
    print(f"\nObjetivo: {num_max} imágenes por clase")
    print(f"Clase con más imágenes: {os.path.basename(carpeta_max)}")
    print("=" * 60)
    
    for i, (nombre, path, num_actual) in enumerate(zip(carpetas, paths, num_archivos)):
        print(f"\n[{i+1}/{len(carpetas)}] {nombre}")
        print(f"  Actuales: {num_actual} | Objetivo: {num_max}")
        
        if num_actual < num_max:
            balancear_clase(path, num_actual, num_max)
        else:
            print(f"  ✓ Ya balanceada")
    
    print("\n" + "=" * 60)
    print("✓ Balanceo completado")

if __name__ == "__main__":
    main()