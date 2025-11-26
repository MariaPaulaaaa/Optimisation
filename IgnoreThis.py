# ============================= PAULA ========================================
import os
import sys

# === CORRECCIÓN DE RUTAS ROBUSTA PARA WINDOWS/ANACONDA ===
# Definimos la ruta base de tu instalación de Anaconda
anaconda_base = r"C:\Users\Maria Paula\anaconda3"

# Lista de carpetas críticas donde Anaconda guarda ejecutables y DLLs
paths_to_add = [
    os.path.join(anaconda_base, "Library", "bin"),        # Aquí están ipopt.exe y glpsol.exe
    os.path.join(anaconda_base, "Library", "mingw-w64", "bin"), # Aquí suelen estar las DLLs de Fortran/C++ faltantes
    os.path.join(anaconda_base, "Scripts"),               # Scripts de Python
    anaconda_base                                         # Raíz (a veces necesaria)
]

# Agregar todas estas rutas al PATH del sistema temporalmente
for path in paths_to_add:
    if os.path.exists(path):
        os.environ['PATH'] += os.pathsep + path
    else:
        print(f"Advertencia: No se encontró la ruta {path}")

print("--- Rutas de Anaconda y Dependencias (DLLs) Agregadas ---")
# ===================================================================