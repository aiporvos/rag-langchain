# 🤖 AI Document Chatbot - Motor RAG

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-orange.svg)](https://langchain.com/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-green.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Un sistema avanzado de **Generación Aumentada por Recuperación (RAG)** construido con LangChain, OpenAI y ChromaDB. Esta aplicación te permite subir múltiples documentos PDF, procesarlos en una base de datos vectorial e interactuar con ellos a través de una interfaz web moderna.

---

## ✨ Características Principales

- 📄 **Soporte Multi-PDF** - Sube y procesa varios documentos simultáneamente.
- ✂️ **Chunking Inteligente** - División de texto optimizada (1000 caracteres con 200 de solapamiento) para una mejor retención del contexto.
- 🔍 **Búsqueda Semántica** - Búsqueda vectorial impulsada por **OpenAI Embeddings** y **ChromaDB**.
- 💬 **Chat Contextual** - IA conversacional usando **GPT-4o-mini** con memoria completa del historial de chat.
- 📊 **Visualización de Embeddings** - Proyección interactiva 2D t-SNE de los fragmentos de tus documentos usando **Plotly**.
- 📚 **Rastreo de Fuentes** - Cada respuesta incluye referencias a los documentos específicos utilizados.
- 🎨 **Interfaz Moderna** - Interfaz elegante y responsiva construida con **Gradio**.

---

## 🛠️ Stack Tecnológico

- **Framework**: [LangChain](https://github.com/hwchase17/langchain)
- **LLM**: OpenAI `gpt-4o-mini`
- **Base de Datos Vectorial**: [ChromaDB](https://www.trychroma.com/)
- **Embeddings**: OpenAI Embeddings
- **Interfaz**: [Gradio](https://gradio.app/)
- **Visualizaciones**: [Plotly](https://plotly.com/) & [Scikit-learn](https://scikit-learn.org/)

---
<img width="1835" height="816" alt="image" src="https://github.com/user-attachments/assets/34e742bb-8647-495f-a969-1d49aa72e887" />

<img width="1833" height="845" alt="image" src="https://github.com/user-attachments/assets/8591de80-efad-4f54-b31c-88d296988b00" />

<img width="1831" height="903" alt="image" src="https://github.com/user-attachments/assets/feb40d72-8297-4422-89d4-2cc7418d404c" />






---

## 🚀 Comenzando

### Requisitos Previos

- Python 3.9+
- Clave de API de OpenAI (OpenAI API Key)

### Instalación

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/aiporvos/rag-langchain.git
   cd rag-langchain
   ```

2. **Configurar un entorno virtual**
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configuración del Entorno**
   Crea un archivo `.env` en el directorio raíz:
   ```env
   OPENAI_API_KEY=tu_clave_de_api_aqui
   ```

### Ejecución Local

```bash
python app.py
```
La aplicación estará disponible en `http://localhost:7860`.

---

## 🏗️ Arquitectura del Sistema

El sistema sigue un pipeline clásico de RAG:

1. **Ingestión**: Los documentos se cargan, se dividen en fragmentos (chunks) y se vectorizan.
2. **Almacenamiento**: Los vectores se almacenan en una instancia local de ChromaDB.
3. **Recuperación**: Cuando un usuario hace una pregunta, el sistema encuentra los fragmentos más relevantes.
4. **Generación**: El LLM sintetiza una respuesta utilizando el contexto recuperado y el historial de la conversación.

---

## 📊 Visualización de Datos

Una característica única de este proyecto es la **Visualización de Embeddings**. Utiliza **t-SNE** (t-Distributed Stochastic Neighbor Embedding) para proyectar el espacio vectorial de alta dimensión en 2D, permitiéndote ver cómo se distribuyen y agrupan los fragmentos de tus documentos.

---

## 📤 Despliegue

Este proyecto está optimizado para **HuggingFace Spaces**. Para desplegar:

1. Crea un nuevo "Space" en HuggingFace.
2. Selecciona **Gradio** como el SDK.
3. Sube `app.py`, `requirements.txt` y `.gitignore`.
4. Agrega tu `OPENAI_API_KEY` en los **Secrets** del Space.

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT; consulta el archivo [LICENSE](LICENSE) para más detalles.

---

Creado con ❤️ como parte del curso de **LLM Engineering**.
