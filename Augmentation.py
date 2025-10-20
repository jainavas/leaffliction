import os
import sys
import cv2
import numpy as np

def save_imgs(flip, rotation, crop, shear, skew, distortion, path_imagen):
    # Extraer directorio
    directorio = os.path.dirname(path_imagen)
    if directorio == "":
        directorio = "."  # FIX: usar directorio actual si está vacío

    # Extraer nombre completo del archivo
    nombre_completo = os.path.basename(path_imagen)

    # Separar nombre y extensión
    nombre_sin_extension, extension = os.path.splitext(nombre_completo)

    # Generar nombres
    nombre_flip = f"{nombre_sin_extension}_Flip{extension}"
    nombre_rotate = f"{nombre_sin_extension}_Rotate{extension}"
    nombre_skew = f"{nombre_sin_extension}_Skew{extension}"
    nombre_shear = f"{nombre_sin_extension}_Shear{extension}"
    nombre_crop = f"{nombre_sin_extension}_Crop{extension}"
    nombre_distortion = f"{nombre_sin_extension}_Distortion{extension}"

    # Generar paths completos
    path_flip = os.path.join(directorio, nombre_flip)
    path_rotate = os.path.join(directorio, nombre_rotate)
    path_skew = os.path.join(directorio, nombre_skew)
    path_shear = os.path.join(directorio, nombre_shear)
    path_crop = os.path.join(directorio, nombre_crop)
    path_distortion = os.path.join(directorio, nombre_distortion)

    # Guardar imágenes
    cv2.imwrite(path_flip, flip)
    cv2.imwrite(path_rotate, rotation)
    cv2.imwrite(path_skew, skew)
    cv2.imwrite(path_shear, shear)
    cv2.imwrite(path_crop, crop)
    cv2.imwrite(path_distortion, distortion)
    
    print(f"6 imágenes guardadas en: {directorio}")

def generate_images(img_path, save):
    img = cv2.imread(img_path)
    
    if img is None:  # FIX: verificar que se leyó correctamente
        print(f"Error: No se pudo leer la imagen {img_path}")
        sys.exit(1)
    
    altura, anchura = img.shape[:2]

    # Flip
    flip_h = cv2.flip(img, 1)
    

    # Rotate
    centro = (anchura // 2, altura // 2)  # FIX: estaba al revés
    angulo = 35
    escala = 1.0
    matriz_rotacion = cv2.getRotationMatrix2D(centro, angulo, escala)
    rotada = cv2.warpAffine(img, matriz_rotacion, (anchura, altura))

    # Crop
    y_inicio = 50
    y_fin = altura - 50
    x_inicio = 50
    x_fin = anchura - 50
    crop = img[y_inicio:y_fin, x_inicio:x_fin]

    # Shear
    shear_factor = 0.2
    matriz_shear = np.array([
        [1, shear_factor, 0],
        [0, 1, 0]  # FIX: la segunda fila debe ser [0, 1, 0]
    ], dtype=np.float32)
    sheared = cv2.warpAffine(img, matriz_shear, (anchura, altura))

    # Skew
    pts_origen = np.float32([
        [0, 0],
        [anchura, 0],
        [0, altura],
        [anchura, altura]
    ])
    offset = 50
    pts_destino = np.float32([
        [offset * 2, 0],
        [anchura - offset, 0],
        [0, altura],
        [anchura, altura]
    ])
    matriz = cv2.getPerspectiveTransform(pts_origen, pts_destino)
    skewed = cv2.warpPerspective(img, matriz, (anchura, altura))

    # Blur/Distortion
    blur = cv2.GaussianBlur(img, (15, 15), 0)


    if save == 0:
        cv2.imshow("normal", img)
        cv2.imshow("skewed", skewed)
        cv2.imshow("fliped", flip_h)
        cv2.imshow("blur", blur)
        cv2.imshow("sheared", sheared)
        cv2.imshow("crop", crop)
        cv2.imshow("rotada", rotada)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save == 1:
        save_imgs(flip_h, rotada, crop, sheared, skewed, blur, img_path)


def augment(path, mode):
    
    img_path = path

    # FIX: Verificar que existe
    if not os.path.exists(img_path):
        print(f"Error: {img_path} no existe")
        sys.exit(1)

    if not os.path.isfile(img_path):
        print(f"Error: {img_path} no es un archivo")
        sys.exit(1)

    directorio = os.path.dirname(img_path)
    if directorio == "":
        directorio = "."
    
    if not os.access(directorio, os.W_OK):
        print(f"Error: No tienes permisos de escritura en {directorio}")
        sys.exit(1)
    
    try:
        save_mode = int(mode)
        if save_mode not in [0, 1]:
            raise ValueError
    except ValueError:
        print("Error: el modo debe ser 0 o 1")
        sys.exit(1)
    
    generate_images(img_path, save_mode)
    

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 Augmentation.py <imagen> <modo>")
        print("  modo: 1 para guardar, 0 solo mostrar")
        sys.exit(1)
    augment(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()