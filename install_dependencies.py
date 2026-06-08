"""
SonicSearch AI - Script de Instalación Rápida

Este script automatiza la instalación de todas las dependencias del proyecto.
Ejecútalo desde una terminal con Python activado o venv.
"""

import subprocess
import sys


def print_section(title: str) -> None:
    """Imprime un título formateado en el console."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def install_packages(packages: list[str]) -> bool:
    """Instala una lista de paquetes uno por uno.
    
    Args:
        packages: Lista de nombres de paquetes a instalar
        
    Returns:
        True si todos los paquetes se instalaron correctamente, False en caso contrario
    """
    print_section("📦 Instalando Paquetes")
    
    installed_count = 0
    
    for package in packages:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                installed_count += 1
                print(f"✅ {package}")
            else:
                print(f"❌ Error instalando {package}:")
                print(result.stderr)
                
        except Exception as e:
            print(f"⚠️  Ocurrió un error al instalar {package}: {e}")
    
    return installed_count == len(packages)


def main() -> None:
    """Función principal de instalación."""
    # Lista completa de paquetes según requirements.txt
    packages = [
        "flask>=3.0.0",           # Servidor web y API REST
        "numpy>=1.24.0",          # Computación numérica (fusión RRF)
        "pandas>=2.0.0",          # Procesamiento de datos CSV
        "faiss-cpu>=1.7.0",       # Búsqueda vectorial acústica (CLAP embeddings)
        
        "langchain-groq>=0.1.0",  # Cliente oficial para Groq LLMs
        "langchain-core>=0.1.0",  # Prompts y parsers JSON
        
        "requests>=2.31.0",       # Descarga de prompts desde Gist
        "einops>=0.6.0"           # Transformaciones tensoriales (fusión RRF)
    ]
    
    print_section("🚀 SonicSearch AI - Instalador de Dependencias")
    print(f"Detectado: {sys.executable}")
    print(f"Paquetes a instalar ({len(packages)}):")
    for pkg in packages:
        print(f"  • {pkg}")
    
    # Instalar todos los paquetes
    success = install_packages(packages)
    
    if success:
        print_section("✅ Instalación Exitosa!")
        print("\nSiguientes pasos:")
        print("1. Ejecuta 'python app.py' para iniciar el servidor")
        print("2. Visita http://localhost:5000 en tu navegador")
        
    else:
        print_section("❗ Instalación Parcial o Fallida")
        print("\nIntenta con:")
        print("  pip install -r requirements.txt")


if __name__ == "__main__":
    main()
