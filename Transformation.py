import cv2
import sys
import os
import matplotlib.pyplot as plt # type: ignore
import numpy as np
import argparse

def parse_arguments():
    """
    Parsea argumentos de línea de comandos.
    """
    parser = argparse.ArgumentParser(
        description='Aplica transformaciones de análisis a imágenes de hojas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Mostrar transformaciones de una imagen
  python3 Transformation.py imagen.jpg
  
  # Procesar directorio y guardar todas las transformaciones
  python3 Transformation.py -src ./Apple/apple_healthy/ -dst ./output/
  
  # Procesar directorio y guardar solo algunas transformaciones
  python3 Transformation.py -src ./Apple/apple_healthy/ -dst ./output/ -mask -edges
        """
    )
    
    # Argumentos posicionales (caso simple)
    parser.add_argument('image', nargs='?', help='Path de la imagen a procesar')
    
    # Argumentos opcionales (caso complejo)
    parser.add_argument('-src', '--source', help='Directorio de origen con imágenes')
    parser.add_argument('-dst', '--destination', help='Directorio de destino para guardar')
    
    # Flags para seleccionar qué transformaciones guardar
    parser.add_argument('-blur', action='store_true', help='Guardar Gaussian Blur')
    parser.add_argument('-mask', action='store_true', help='Guardar Mask')
    parser.add_argument('-edges', action='store_true', help='Guardar Edge Detection')
    parser.add_argument('-roi', action='store_true', help='Guardar ROI Objects')
    parser.add_argument('-analyze', action='store_true', help='Guardar Analyze Object')
    parser.add_argument('-landmarks', action='store_true', help='Guardar Pseudolandmarks')
    parser.add_argument('-texture', action='store_true', help='Guardar Texture Analysis')
    parser.add_argument('-histogram', action='store_true', help='Guardar Color Histogram')
    parser.add_argument('-all', action='store_true', help='Guardar todas las transformaciones')
    
    return parser.parse_args()

def process_and_save(img_path, dst_dir, transformations):
    """
    Procesa una imagen y guarda las transformaciones seleccionadas.
    """
    # Leer imagen
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen")
    
    # Extraer nombre base
    nombre_base = os.path.splitext(os.path.basename(img_path))[0]
    
    # Aplicar transformaciones
    blur = gaussian_blur(img)
    mask = create_mask(img)
    edges = edge_detection(img)
    roi_img, contours = find_roi_objects(img, mask)
    analyzed, info = analyze_object(img, contours)
    landmarks = pseudolandmarks(img, contours)
    texture = texture_analysis(img)
    
    # Guardar según flags
    saved_count = 0
    
    if transformations['blur']:
        path = os.path.join(dst_dir, f"{nombre_base}_blur.jpg")
        cv2.imwrite(path, blur)
        saved_count += 1
    
    if transformations['mask']:
        path = os.path.join(dst_dir, f"{nombre_base}_mask.jpg")
        cv2.imwrite(path, mask)
        saved_count += 1
    
    if transformations['edges']:
        path = os.path.join(dst_dir, f"{nombre_base}_edges.jpg")
        cv2.imwrite(path, edges)
        saved_count += 1
    
    if transformations['roi']:
        path = os.path.join(dst_dir, f"{nombre_base}_roi.jpg")
        cv2.imwrite(path, roi_img)
        saved_count += 1
    
    if transformations['analyze']:
        path = os.path.join(dst_dir, f"{nombre_base}_analyzed.jpg")
        cv2.imwrite(path, analyzed)
        saved_count += 1
    
    if transformations['landmarks']:
        path = os.path.join(dst_dir, f"{nombre_base}_landmarks.jpg")
        cv2.imwrite(path, landmarks)
        saved_count += 1
    
    if transformations['texture']:
        path = os.path.join(dst_dir, f"{nombre_base}_texture.jpg")
        cv2.imwrite(path, texture)
        saved_count += 1
    
    if transformations['histogram']:
        path = os.path.join(dst_dir, f"{nombre_base}_histogram.png")
        fig = color_histogram_plant(img, mask)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_count += 1
    
    print(f"  ✓ Guardadas {saved_count} transformaciones")

def process_directory(src_dir, dst_dir, transformations):
    """
    Procesa todas las imágenes de un directorio.
    
    Args:
        src_dir: Directorio origen
        dst_dir: Directorio destino
        transformations: Dict con flags de qué transformaciones guardar
    """
    # Verificar que existe el directorio origen
    if not os.path.exists(src_dir):
        print(f"Error: {src_dir} no existe")
        sys.exit(1)
    
    if not os.path.isdir(src_dir):
        print(f"Error: {src_dir} no es un directorio")
        sys.exit(1)
    
    # Crear directorio destino si no existe
    os.makedirs(dst_dir, exist_ok=True)
    
    # Listar imágenes
    extensiones = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG')
    imagenes = [f for f in os.listdir(src_dir) if f.endswith(extensiones)]
    
    if not imagenes:
        print(f"No se encontraron imágenes en {src_dir}")
        return
    
    print(f"Procesando {len(imagenes)} imágenes de {src_dir}...")
    print(f"Guardando resultados en {dst_dir}")
    print(f"Transformaciones a guardar: {[k for k, v in transformations.items() if v]}")
    print("-" * 60)
    
    # Procesar cada imagen
    for i, img_name in enumerate(imagenes, 1):
        img_path = os.path.join(src_dir, img_name)
        
        print(f"[{i}/{len(imagenes)}] Procesando {img_name}...")
        
        try:
            process_and_save(img_path, dst_dir, transformations)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    print("-" * 60)
    print(f"✓ Procesamiento completado")


def gaussian_blur(imagen):
    """
    Aplica desenfoque gaussiano para reducir ruido.
    """
    # Kernel size debe ser impar: (3,3), (5,5), (7,7), (9,9), etc.
    blur = cv2.GaussianBlur(imagen, (5, 5), 0)
    return blur

def create_mask(imagen):
    """
    Crea máscara binaria para segmentar SOLO la hoja.
    """
    # Convertir a HSV
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    
    # Rango de verde amplio (incluye verde-amarillo de hojas enfermas)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    
    # Crear máscara SOLO por color verde
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Operaciones morfológicas para limpiar
    kernel = np.ones((5, 5), np.uint8)
    
    # Cerrar huecos pequeños dentro de la hoja
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Eliminar ruido pequeño fuera de la hoja
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # IMPORTANTE: Quedarse SOLO con el objeto más grande
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Filtrar contornos pequeños (ruido)
        min_area = 1000  # Ajusta según tu dataset
        large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        if large_contours:
            # Tomar SOLO el contorno más grande (la hoja)
            largest_contour = max(large_contours, key=cv2.contourArea)
            
            # Crear máscara limpia con SOLO ese contorno
            mask_clean = np.zeros_like(mask)
            cv2.drawContours(mask_clean, [largest_contour], -1, 255, -1)
            
            return mask_clean
    
    return mask

def find_roi_objects(imagen, mask):
    """
    Encuentra objetos (contornos) en la imagen usando la máscara.
    """
    # Encontrar contornos
    contours, hierarchy = cv2.findContours(
        mask, 
        cv2.RETR_EXTERNAL,  # Solo contornos externos
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Dibujar contornos en la imagen
    img_contours = imagen.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    
    return img_contours, contours

def analyze_object(imagen, contours):
    """
    Analiza propiedades del objeto más grande (la hoja).
    """
    if not contours:
        return imagen, {}
    
    # Tomar el contorno más grande (asumimos que es la hoja)
    contour = max(contours, key=cv2.contourArea)
    
    # Calcular propiedades
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(contour)
    
    # Elipse que mejor ajusta
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        
        # Dibujar
        img_analyzed = imagen.copy()
        cv2.ellipse(img_analyzed, ellipse, (255, 0, 255), 2)
        cv2.rectangle(img_analyzed, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        img_analyzed = imagen.copy()
        cv2.rectangle(img_analyzed, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Info
    info = {
        'area': area,
        'perimeter': perimeter,
        'width': w,
        'height': h,
        'aspect_ratio': float(w)/h if h != 0 else 0
    }
    
    return img_analyzed, info

def pseudolandmarks(imagen, contours, num_points=20):
    """
    Divide el contorno en puntos equidistantes.
    """
    if not contours:
        return imagen
    
    contour = max(contours, key=cv2.contourArea)
    
    # Aproximar contorno para suavizar
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Tomar puntos equidistantes
    total_points = len(contour)
    step = max(1, total_points // num_points)
    landmarks = contour[::step]
    
    # Dibujar
    img_landmarks = imagen.copy()
    for point in landmarks:
        x, y = point[0]
        cv2.circle(img_landmarks, (x, y), 5, (255, 0, 0), -1)
    
    # Dibujar líneas entre landmarks
    for i in range(len(landmarks)):
        pt1 = tuple(landmarks[i][0])
        pt2 = tuple(landmarks[(i+1) % len(landmarks)][0])
        cv2.line(img_landmarks, pt1, pt2, (0, 255, 255), 1)
    
    return img_landmarks

def color_histogram_plant(imagen, mask=None):
    """
    Histograma enfocado en análisis de salud de plantas.
    """
    # Aplicar máscara
    if mask is not None:
        imagen_masked = cv2.bitwise_and(imagen, imagen, mask=mask)
    else:
        imagen_masked = imagen
    
    # Convertir a HSV
    hsv = cv2.cvtColor(imagen_masked, cv2.COLOR_BGR2HSV)
    
    # Crear figura
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Plant Color Analysis', fontsize=16)
    
    # 1. Hue (color) - importante para detectar amarillamiento
    hue = hsv[:, :, 0]
    if mask is not None:
        hue = hue[mask > 0]
    axes[0, 0].hist(hue.flatten(), bins=180, range=[0, 180], color='green', alpha=0.7)
    axes[0, 0].set_title('Hue Distribution (Green=Healthy)')
    axes[0, 0].set_xlabel('Hue Value')
    axes[0, 0].axvline(x=60, color='r', linestyle='--', label='Green center')
    axes[0, 0].legend()
    
    # 2. Saturation (intensidad del color)
    sat = hsv[:, :, 1]
    if mask is not None:
        sat = sat[mask > 0]
    axes[0, 1].hist(sat.flatten(), bins=256, range=[0, 256], color='cyan', alpha=0.7)
    axes[0, 1].set_title('Saturation (Color Intensity)')
    axes[0, 1].set_xlabel('Saturation Value')
    
    # 3. Value (brillo)
    val = hsv[:, :, 2]
    if mask is not None:
        val = val[mask > 0]
    axes[1, 0].hist(val.flatten(), bins=256, range=[0, 256], color='yellow', alpha=0.7)
    axes[1, 0].set_title('Value (Brightness)')
    axes[1, 0].set_xlabel('Value')
    
    # 4. Green channel (canal verde específico)
    green = imagen_masked[:, :, 1]
    if mask is not None:
        green = green[mask > 0]
    axes[1, 1].hist(green.flatten(), bins=256, range=[0, 256], color='lime', alpha=0.7)
    axes[1, 1].set_title('Green Channel')
    axes[1, 1].set_xlabel('Green Intensity')
    
    # Calcular métricas
    mean_hue = np.mean(hue)
    mean_sat = np.mean(sat)
    mean_val = np.mean(val)
    
    fig.text(0.5, 0.02, 
             f'Mean Hue: {mean_hue:.1f} | Mean Saturation: {mean_sat:.1f} | Mean Value: {mean_val:.1f}',
             ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig

def process_single_image(img_path):
    """
    Procesa una sola imagen y muestra todas las transformaciones.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: no se pudo leer {img_path}")
        return
    
    # Aplicar TODAS las transformaciones
    blur = gaussian_blur(img)
    mask = create_mask(img)
    edges = edge_detection(img)  # NUEVA
    roi_img, contours = find_roi_objects(img, mask)
    analyzed, info = analyze_object(img, contours)
    landmarks = pseudolandmarks(img, contours)
    texture = texture_analysis(img)  # NUEVA
    
    # Mostrar en grid 3x3 (para que quepan todas)
    plt.figure(figsize=(18, 12))
    
    plt.subplot(3, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(3, 3, 2)
    plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
    plt.title('Gaussian Blur')
    plt.axis('off')
    
    plt.subplot(3, 3, 3)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.axis('off')
    
    plt.subplot(3, 3, 4)
    plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))  # NUEVA
    plt.title('Edge Detection')
    plt.axis('off')
    
    plt.subplot(3, 3, 5)
    plt.imshow(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
    plt.title('ROI Objects')
    plt.axis('off')
    
    plt.subplot(3, 3, 6)
    plt.imshow(cv2.cvtColor(analyzed, cv2.COLOR_BGR2RGB))
    plt.title('Analyze Object')
    plt.axis('off')
    
    plt.subplot(3, 3, 7)
    plt.imshow(cv2.cvtColor(landmarks, cv2.COLOR_BGR2RGB))
    plt.title('Pseudolandmarks')
    plt.axis('off')
    
    plt.subplot(3, 3, 8)
    plt.imshow(cv2.cvtColor(texture, cv2.COLOR_BGR2RGB))  # NUEVA
    plt.title('Texture Analysis')
    plt.axis('off')
    
    # Dejar el último espacio para info o vacío
    plt.subplot(3, 3, 9)
    plt.axis('off')
    if info:
        info_text = f"Area: {info['area']:.0f}px²\n"
        info_text += f"Perimeter: {info['perimeter']:.0f}px\n"
        info_text += f"Width: {info['width']}px\n"
        info_text += f"Height: {info['height']}px\n"
        info_text += f"Aspect Ratio: {info['aspect_ratio']:.2f}"
        plt.text(0.1, 0.5, info_text, fontsize=12, 
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show(block=False)
    
    # Histograma en ventana separada
    color_histogram_plant(img, mask)
    plt.show()

def texture_analysis(imagen):
    """
    Analiza la textura de la hoja usando filtros Gabor o LBP.
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Aplicar filtro Sobel para detectar textura
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitud del gradiente
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = np.uint8(magnitude / magnitude.max() * 255)
    
    # Aplicar colormap para visualizar mejor
    texture_colored = cv2.applyColorMap(magnitude, cv2.COLORMAP_JET)
    
    return texture_colored

def edge_detection(imagen):
    """
    Detecta bordes usando Canny.
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Aplicar blur para reducir ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detección de bordes Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Convertir a BGR para visualizar
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    return edges_colored

def main():
    args = parse_arguments()
    
    # Caso 1: Una sola imagen (caso simple)
    if args.image and not args.source:
        process_single_image(args.image)
        return
    
    # Caso 2: Directorio (caso complejo)
    if args.source and args.destination:
        # Determinar qué transformaciones guardar
        transformations = {
            'blur': args.blur or args.all,
            'mask': args.mask or args.all,
            'edges': args.edges or args.all,
            'roi': args.roi or args.all,
            'analyze': args.analyze or args.all,
            'landmarks': args.landmarks or args.all,
            'texture': args.texture or args.all,
            'histogram': args.histogram or args.all,
        }
        
        # Si no se especificó ninguna, guardar todas
        if not any(transformations.values()):
            print("No se especificaron transformaciones, guardando todas...")
            transformations = {k: True for k in transformations}
        
        process_directory(args.source, args.destination, transformations)
        return
    
    # Caso 3: Argumentos inválidos
    print("Error: Argumentos inválidos")
    print("\nUso:")
    print("  python3 Transformation.py <imagen>")
    print("  python3 Transformation.py -src <directorio> -dst <directorio> [-mask] [-edges] ...")
    print("\nPara más ayuda: python3 Transformation.py -h")
    sys.exit(1)

if __name__ == "__main__":
    main()