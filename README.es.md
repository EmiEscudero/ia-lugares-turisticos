# Reconocimiento de lugares turísticos de México

Español · [English](README.md)

Una aplicación de **Streamlit en español e inglés** que propone destinos y estados de México a partir de una foto. Utiliza EfficientNet-B0 preentrenada, entrenamiento con GPU y evaluación por destino. Conserva los cuadernos y la interfaz originales como antecedentes del proyecto.

La interfaz comienza en español. El control **Idioma / Language** permite cambiar a inglés sin perder la foto ni las predicciones. Muestra las cinco mejores coincidencias con porcentajes, indica resultados inciertos y permite descargar los resultados. Las puntuaciones comparan clases conocidas; no garantizan que el destino sea correcto.

## Resultados del piloto

Se revisaron las **4,668 imágenes** de Drive en una copia local. Después de agrupar duplicados, originales y reflejos, excluir familias con etiquetas contradictorias y filtrar destinos con muy pocas fotos, el piloto incluye **164 destinos** y **1,375 fotos**: 957 para entrenamiento, 209 para validación y 209 para prueba.

| Modelo | Primera opción correcta | Destino entre las cinco opciones |
| --- | ---: | ---: |
| CNN original adaptada al mismo experimento | 1.9 % | 11.5 % |
| EfficientNet seleccionada con validación | **31.6 %** | **54.5 %** |

Son resultados exploratorios sobre las mismas 209 fotos reservadas. El modelo no cubre los 480 destinos del conjunto original. Los porcentajes originales de playa/pueblo corresponden a otra tarea y no son comparables. Consulta [la revisión del proyecto](docs/REVIEW.md) y [los experimentos completos](docs/EXPERIMENTS.md) para conocer las limitaciones y los resultados de las cuatro comparaciones.

En la RTX 4060 de 8 GB, el ajuste fino estable tardó una mediana de **6.6 segundos por época** después del arranque, con aproximadamente **1.43 GiB de memoria de GPU asignada** y lote de 32. No se midió la laptop anterior, por lo que no se afirma un factor de aceleración.

## Ejecutar en Windows

Desde la carpeta del proyecto, prepara el entorno si todavía no existe:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Se probó con Python 3.13.2. Para otra plataforma, consulta el [instalador oficial de PyTorch](https://pytorch.org/get-started/locally/).

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Abre [la aplicación local](http://127.0.0.1:8501/?lang=es) o [la versión en inglés](http://127.0.0.1:8501/?lang=en). Elige una foto JPEG, PNG o WebP, de hasta 20 MB y 30 megapíxeles, y pulsa **Encontrar este lugar**. La aplicación procesa las fotos en la memoria del equipo y no las guarda.

El código busca modelos en `runs/<nombre>/best.pt` y da prioridad a `runs/recommended/best.pt`. Los modelos y las fotos se mantienen locales y no se incluyen en Git: clonar el repositorio no descarga un modelo entrenado. La copia de trabajo de este equipo ya contiene el modelo seleccionado.

## Datos y entrenamiento

La carpeta original de Drive se usó **solo para lectura**. La preparación y el entrenamiento operan sobre la copia local; ninguna foto original se borró, movió, renombró ni modificó. Los reportes indican qué destinos se excluyeron del piloto y por qué.

Las instrucciones detalladas para preparar nuevas imágenes, entrenar, evaluar y usar la línea de comandos están en [README.md](README.md). El [experimento guardado](experiments/pilot/summary.json) incluye la configuración y las métricas; su manifest conserva la división exacta de las imágenes.

El siguiente paso de mayor impacto es recopilar más fotos independientes y variadas, revisar destinos que se solapan y medir el comportamiento con lugares desconocidos. Las playas genéricas pueden no contener información suficiente para identificar un destino concreto.
