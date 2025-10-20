import os
import sys
import matplotlib.pyplot as plt # type: ignore

if len(sys.argv) != 2:
	print("Usage: python3 Distribution.py 'filter by name'\n")
	sys.exit()

filter = sys.argv[1].lower()
ruta = os.curdir


carpetas = []
num_archivos = []
max_num_archivos = -1

for carpeta, subcarpetas, archivos in os.walk(ruta):
	if filter in os.path.basename(carpeta).lower():
		carpetas.append(os.path.basename(carpeta))
		num_archivos.append(len(archivos))
		if max_num_archivos < len(archivos):
			carpeta_max = carpeta
			max_num_archivos = len(archivos)


if not carpetas:
    print(f"No se encontraron carpetas con '{filter}' en el nombre.")
    sys.exit(0)

print(f"La carpeta mas grande es {carpeta_max} con {max_num_archivos}\n")

plt.figure(figsize=(10, 5))
plt.bar(carpetas, num_archivos)
plt.title(f"Número de archivos por carpeta que contiene '{filter}'")
plt.xlabel("Carpeta")
plt.ylabel("Cantidad de archivos")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show(block=False)

plt.figure(figsize=(7, 7))
plt.pie(num_archivos, labels=carpetas, autopct="%1.1f%%", startangle=140)
plt.title(f"Distribución de archivos por carpeta ('{filter}')")
plt.tight_layout()
plt.show()