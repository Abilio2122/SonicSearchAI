# SonicSearch AI - Instalación de Dependencias

## Requisitos Previos

- **Python 3.10 o superior** instalado en tu sistema
- **pip** (package manager de Python) disponible
- Un editor de código como VS Code, PyCharm o similar

## Pasos para Instalar las Dependencias

### Opción 1: Usar pip directamente (Recomendado)

```bash
# Navega al directorio del proyecto
cd C:\Users\user\OneDrive\Desktop\LLM\SonicSearchAI

# Instala todas las dependencias desde requirements.txt
pip install -r requirements.txt
```

### Opción 2: Instalar paquetes uno por uno

Si prefieres instalar los paquetes individualmente, ejecuta estos comandos en orden:

```bash
# Servidor web y API REST
pip install flask>=3.0.0

# Computación numérica (fusión RRF)
pip install numpy>=1.24.0

# Procesamiento de datos CSV
pip install pandas>=2.0.0

# Búsqueda vectorial acústica (CLAP embeddings)
pip install faiss-cpu>=1.7.0

# LangChain + Groq API
pip install langchain-groq>=0.1.0
pip install langchain-core>=0.1.0

# Utilidades adicionales
pip install requests>=2.31.0
pip install einops>=0.6.0
```

### Opción 3: Usar venv (Ambiente Virtual)

Para un entorno aislado y limpio de dependencias:

```bash
# Crear ambiente virtual
python -m venv .venv

# Activar el ambiente virtual en Windows
.\.venv\Scripts\activate

# Instalar las dependencias desde requirements.txt
pip install -r requirements.txt
```

## Verificación de Instalación

Después de instalar todas las dependencias, verifica que todo funcione correctamente ejecutando:

```bash
python --version  # Debería mostrar Python 3.10 o superior
pip list           # Lista todos los paquetes instalados
```

## Configuración del Archivo `.env` ⚙️

**Importante**: Para usar las dependencias de LangChain con Groq, necesitas configurar tu API Key en un archivo `.env`.

### Paso Adicional: Configurar Variables de Entorno

1. **Copia el archivo de ejemplo:**
   ```bash
   copy .env.example .env
   ```

2. **Edita el archivo `.env` y agrega tu clave:**
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   # Opcional: FLASK_SECRET_KEY=my_secret_key_1234567890abcdef
   ```

3. **Obtén tu API Key de Groq:**
   - Ve a https://console.groq.com/keys
   - Crea una nueva clave o usa la existente
   
⚠️ **Seguridad**: 
- Nunca compartas tu `GROQ_API_KEY` en repositorios públicos
- El archivo `.env` está excluido de Git (ver `.gitignore`)

## Notas Importantes

- **faiss-cpu**: Se usa la versión CPU (no GPU) para compatibilidad máxima con Windows
- **.venv/**: El directorio de ambiente virtual se incluye en `.gitignore` y no debe ser commitado al repositorio
- **Caché Python**: Los archivos `__pycache__/` también están excluidos del control de versiones

## Problemas Comunes

### "ModuleNotFoundError: No module named 'flask'"
**Solución**: Asegúrate de haber ejecutado `pip install -r requirements.txt`. Si usaste venv, activa el ambiente primero.

### "Could not find a version that satisfies the requirement"
**Solución**: Verifica que tu versión de Python sea compatible (3.10+). Algunos paquetes requieren versiones específicas del interprete.

## Recursos Adicionles

- [Documentación oficial de Flask](https://flask.palletsprojects.com/)
- [LangChain Documentation](https://docs.langchain.com/)
- [FAISS GitHub Repository](https://github.com/facebookresearch/faiss)
