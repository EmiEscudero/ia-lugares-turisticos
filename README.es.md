# Reconocimiento de lugares turísticos de México

Español · [English](README.md)

Identifica un **destino y su estado** a partir de una fotografía. El nuevo flujo de `tourist_places` utiliza EfficientNet-B0 preentrenada, etiquetas unificadas por destino, divisiones de entrenamiento, validación y prueba que mantienen juntos los duplicados, y precisión mixta con CUDA. Los cuadernos y la interfaz originales se conservan como experimentos históricos; sus archivos de pesos no son compatibles con el nuevo modelo.

Consulta [la revisión del proyecto](docs/REVIEW.md) antes de interpretar las cifras originales de exactitud: miden la clasificación de playa frente a pueblo, no el acierto del lugar concreto.

El [piloto con datos reales](docs/EXPERIMENTS.md) cubre **164 destinos** después de agrupar duplicados y apartar familias con etiquetas contradictorias. El modelo seleccionado alcanza **31,6 % de aciertos en la primera opción y 54,5 % entre las cinco primeras** en prueba, frente a **1,9 % y 11,5 %** de una adaptación de la CNN original sobre las mismas 209 fotos de prueba. Es una comparación exploratoria; no cubre los 480 destinos de origen. Las fotos de entrenamiento permanecen privadas y los pesos entrenados están excluidos de Git: clonar el código no descarga un modelo entrenado. El despliegue preparado para la nube obtiene los modelos sin datos privados de una publicación de GitHub.

## Preparación del entorno en Windows / RTX 4060

Ejecuta PowerShell desde la carpeta del proyecto. Se utilizó Python 3.13 para la validación. Las dependencias se guardan en `.venv` y los pesos descargados, en `artifacts/`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

El último comando debe mostrar `True` y el nombre de la GPU NVIDIA. Para otra plataforma, selecciona la versión adecuada en el [instalador oficial de PyTorch](https://pytorch.org/get-started/locally/). Las dos versiones fijadas arriba corresponden al par con CUDA instalado para este proyecto; `requirements.txt` indica el intervalo de compatibilidad de las API.

## Preparar las imágenes

Proporciona una copia local autorizada de la colección de imágenes de entrenamiento. La documentación actual no publica enlaces de descarga privados. Eliminar esos enlaces no cambia los permisos de acceso de Drive; los commits anteriores de Git pueden conservar enlaces previos. Coloca las dos carpetas de categorías dentro de una misma carpeta de origen, por ejemplo:

```text
data/raw/
  PLAYAS/
    QUINTANA_ROO/
      TULUM/
        train/*.jpg
        validation/*.jpg
  PUEBLOS/
    JALISCO/
      TEQUILA/
        train/*.jpg
        validation/*.jpg
```

La carpeta de origen puede estar en otra ubicación del disco; indícala mediante `--source`. La preparación lee las imágenes donde están y nunca las modifica ni las copia. Se reúnen las carpetas antiguas `train` y `validation` y se crea una nueva división, porque la original contiene fotos duplicadas entre ambos conjuntos. Los CSV históricos **no** se utilizan para asignar las nuevas etiquetas.

```powershell
.\.venv\Scripts\python.exe -m tourist_places.audit --output artifacts/legacy_audit.json
.\.venv\Scripts\python.exe -m tourist_places.prepare --source data/raw --output data/prepared --min-images 6
```

Revisa `data/prepared/preparation_report.json` antes de entrenar:

- Las etiquetas se obtienen de los nombres normalizados de las carpetas `(estado, destino)`. Un lugar presente en ambas categorías se convierte en un solo destino con las dos etiquetas de tipo. Los nombres y límites geográficos aún requieren revisión humana: una región como Riviera Maya se solapa con destinos más concretos.
- Una misma familia de imágenes no puede tener etiquetas de destinos contradictorias; la preparación se detiene y registra los conflictos para corregirlos. En un experimento exploratorio, `--exclude-conflicts` aparta explícitamente todas las imágenes de esas familias, sin suponer cuál es la etiqueta correcta. No se borra ni se modifica ninguna imagen de origen.
- Los duplicados exactos y las fotos similares recomprimidas o reflejadas se agrupan en toda la colección antes de dividirla. Se conserva un representante por familia sin conflictos. La comparación aproximada utiliza un hash de diferencias de 64 bits y una distancia predeterminada de 4; revisa los grupos para detectar coincidencias accidentales, especialmente entre horizontes genéricos. Usa `--hash-distance -1` para desactivar la comparación aproximada y conservar las comprobaciones exactas del contenido decodificado. Los recortes y la independencia de las fuentes aún requieren revisión humana.
- Los destinos con menos de 6 grupos independientes se excluyen y se enumeran en el informe. Es un filtro para el piloto, no una afirmación de que 6 fotos basten para reconocer un lugar de forma fiable. Recopila más fotos independientes antes de ampliar la cobertura. Las copias reflejadas no cuentan como evidencia nueva.
- Cada destino incluido recibe aproximadamente un 70 % de imágenes para entrenamiento, un 15 % para validación y un 15 % para prueba, con al menos una imagen en cada conjunto. Se guardan la semilla y el resumen criptográfico del manifest para identificar la división. No se puede sobrescribir un manifest existente.
- Para una evaluación más sólida, agrupa manualmente las fotos del mismo fotógrafo, visita o fuente antes de crear un conjunto de referencia definitivo. La detección visual de duplicados no permite establecer la independencia de las fuentes.

Si la preparación excluye la mayoría de los destinos, empieza con un conjunto más pequeño y bien fotografiado y recopila más fotos. Como objetivo inicial, busca entre 50 y 100 fotos independientes por destino, con variedad de puntos de vista, clima, iluminación y cámaras. La cantidad necesaria depende de lo reconocible que sea cada lugar; una foto genérica de arena y agua puede no contener información suficiente para identificar una playa concreta.

## Entrenar y comparar

```powershell
.\.venv\Scripts\python.exe -m tourist_places.train --device cuda --prepared data/prepared --output runs/first --batch-size 16 --epochs 25
```

Por defecto, el entrenamiento ajusta el nuevo clasificador durante 3 épocas con la red preentrenada congelada y después realiza un ajuste fino de esa red con una tasa de aprendizaje menor. Utiliza entradas de 224 píxeles, aumento de datos solo durante el entrenamiento, normalización de ImageNet, AdamW, un programa de tasa de aprendizaje y parada temprana. Se selecciona el mejor archivo de pesos según el macro-F1 de validación; en caso de empate, se utiliza la pérdida de validación. Las estadísticas acumuladas de BatchNorm preentrenada permanecen fijas durante el ajuste fino; `--update-batchnorm` habilita una alternativa que se puede comparar con datos de validación.

`history.csv` registra la exactitud top-1 y top-5 por **destino**, macro-F1, macro-recall, pérdida, segundos por época y pico de memoria de GPU asignada. La exactitud se calcula a partir del número de imágenes, sin promediar los porcentajes de los lotes. Cada ejecución guarda su configuración, correspondencia de clases e identidad de la división. Las semillas permiten repetir la división de datos; el entrenamiento con CUDA no garantiza resultados idénticos bit a bit.

Empieza con un tamaño de lote de 16 en la RTX 4060 de 8 GB. Prueba 32 después de medir el uso de memoria; utiliza 8 si es necesario. La carga de datos emplea por defecto dos procesos de trabajo y memoria fijada cuando se usa CUDA. Utiliza `--workers 0` para diagnosticar problemas con los procesos de carga en Windows. `--no-amp` permite comparar con precisión completa. `--no-pretrained` existe para pruebas rápidas de funcionamiento sin conexión y no debe utilizarse como referencia real.

Mide varias épocas después del arranque para establecer la velocidad. El inicio de los procesos de carga añade tiempo a la primera época. El ajuste automático de convoluciones está desactivado por defecto para reducir el arranque en conjuntos pequeños; `--cudnn-benchmark` lo activa y puede añadir un tiempo considerable a la primera época y a la primera época con la red descongelada. Ni el nombre de la GPU ni una prueba rápida con datos sintéticos permiten establecer un factor de aceleración o la exactitud por destino en este conjunto.

Para comparar el diseño de la red original con exactamente las mismas etiquetas de destino y la misma división:

```powershell
.\.venv\Scripts\python.exe -m tourist_places.train --architecture legacy_cnn --device cuda --prepared data/prepared --output runs/legacy_baseline
```

Esta referencia conserva las tres etapas convolucionales originales y la capa densa de 512 unidades, con entradas de 128 píxeles y sin normalización de ImageNet. Sustituye las salidas jerárquicas por una sola salida de destino, utiliza la misma división depurada y aumento de datos solo durante el entrenamiento, y entrena todos los pesos desde cero. Es una referencia controlada basada en el diseño original, no la exactitud de playa/pueblo del cuaderno original.

## Evaluar y predecir

Utiliza la validación para decidir la configuración. Una vez fijados el modelo y el conjunto de datos, evalúa el conjunto de prueba reservado; no ajustes la configuración repetidamente en función de sus resultados.

```powershell
.\.venv\Scripts\python.exe -m tourist_places.evaluate --checkpoint runs/first/best.pt --prepared data/prepared --split val
.\.venv\Scripts\python.exe -m tourist_places.evaluate --checkpoint runs/first/best.pt --prepared data/prepared --split test
.\.venv\Scripts\python.exe -m tourist_places.predict "C:\photos\example.jpg" --checkpoint runs/first/best.pt --top-k 5
```

La evaluación guarda métricas globales, recall por destino y los conteos de confusión distintos de cero. `--data-root` en entrenamiento y evaluación permite cambiar la ubicación de las imágenes de origen sin modificar la división guardada. Las predicciones devuelven el nombre, estado y etiquetas de tipo guardados. Las puntuaciones inferiores a `--min-confidence` (0.5 por defecto) se marcan como inciertas. Este umbral es una regla de presentación: las puntuaciones softmax no son una certeza calibrada y una puntuación alta no demuestra que un lugar desconocido pertenezca a una clase conocida. Valida el comportamiento de rechazo con lugares no vistos antes de publicar la aplicación.

El archivo original `predictGUI.py` sigue cargando el modelo histórico `modeloCNN.pth`. Utiliza la nueva línea de comandos o la interfaz de Streamlit para los nuevos archivos de pesos.

## Interfaz local de Streamlit

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Abre [la aplicación local](http://127.0.0.1:8501). Comienza en **español**; usa **Idioma / Language** en la parte superior para cambiar a inglés. Los enlaces directos pueden seleccionar [español](http://127.0.0.1:8501/?lang=es) o [inglés](http://127.0.0.1:8501/?lang=en). Cambiar de idioma conserva la foto seleccionada, el modelo, el umbral y la predicción.

Sube una foto JPEG, PNG o WebP y selecciona **Encontrar este lugar / Find this place**. La aplicación muestra destinos y estados ordenados por puntuación, porcentajes, coincidencias inciertas, resultados descargables, la calidad medida del modelo y su cobertura de destinos. Los modelos entrenados en `runs/<nombre>/best.pt` se descubren al actualizar la lista. Los modelos de pruebas rápidas se excluyen. Los ejemplos opcionales de `data/ui_examples/` deben proceder del conjunto de entrenamiento o validación, nunca del conjunto de prueba reservado.

Las cuatro variantes del piloto aparecen en un orden fijo, según su **macro-F1 de validación, de mayor a menor**:

| Orden | Modelo | Propósito | Macro-F1 de validación |
| --- | --- | --- | ---: |
| 1 | Ajuste fino estable · `stable_transfer_pilot` | Modelo recomendado y seleccionado por defecto; ajusta EfficientNet manteniendo fijas las estadísticas de BatchNorm. | 0,310 |
| 2 | Características fijas · `frozen_pilot` | Entrena el clasificador y conserva fija la red preentrenada. | 0,287 |
| 3 | Ajuste fino inicial · `transfer_pilot` | Primera comparación de ajuste fino, con actualización de BatchNorm y ajuste automático de convoluciones. | 0,228 |
| 4 | CNN original adaptada · `legacy_pilot` | Referencia entrenada desde cero a partir del diseño de la CNN original. | 0,029 |

Los números indican el **orden de las opciones**, no la exactitud ni el porcentaje de una predicción. `runs/recommended/best.pt` es una copia local del modelo estable seleccionado, no un quinto modelo. Si se verifica que la copia coincide, se utiliza para la primera opción. Otras ejecuciones aparecen después de las variantes conocidas, ordenadas por nombre. El selector aparece en la página principal; **Acerca del modelo** explica cada variante y permite comparar sus resultados. La selección se hizo con validación antes de evaluar la prueba; no se cambia el recomendado para favorecer el mejor resultado observado en prueba.

**Acerca del modelo** también muestra una descripción y dos diagramas que cambian con el modelo seleccionado: cómo se organizaron su entrenamiento y evaluación, y cómo una foto se convierte en predicciones ordenadas. Los diagramas distinguen las características preentrenadas y entradas de 224 píxeles de EfficientNet de las entradas de 128 píxeles y el aprendizaje desde cero de la CNN original. Las cantidades de fotos del piloto solo se muestran cuando el modelo conserva la identidad de su división de datos. Son explicaciones esquemáticas del proceso, no explicaciones visuales de qué píxeles influyeron en una predicción concreta.

El comando local anterior hace que el servidor escuche únicamente en `127.0.0.1`, por lo que esa instancia solo está disponible en este equipo. Las fotos subidas permanecen en memoria y la aplicación no las guarda. Un pequeño componente local proporciona controles accesibles de carga en español e inglés, sin scripts externos ni almacenamiento persistente en el navegador. El servidor también valida el tamaño del archivo, el formato decodificado y el número de píxeles. La inferencia se ejecuta en CPU para dejar la GPU libre para entrenar. La caché del modelo incluye la fecha de modificación de los pesos, de modo que reemplazar un archivo no reutiliza pesos antiguos. La carpeta original de Drive es una fuente privada de datos; la interfaz no tiene capacidad para editarla.

El despliegue en la nube está preparado para utilizar `cloud/streamlit_app.py` desde la rama de desarrollo `dev/improve-place-recognition`, con Python 3.13 y dependencias separadas para CPU. Descarga cuatro modelos sin datos privados de una publicación de GitHub y verifica cada archivo con su resumen SHA-256 fijado. Los pesos permanecen fuera de Git y no se suben fotos de origen para el despliegue. Consulta las [instrucciones de despliegue](docs/DEPLOYMENT.md) para conocer la configuración y los archivos de la publicación.

**No es necesario usar la rama `main`**: Streamlit Community Cloud permite seleccionar el repositorio, la rama y el archivo de entrada, según su [guía oficial de despliegue](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy). El comando local anterior sigue sirviendo la aplicación únicamente en localhost; ejecutarlo no la publica ni envía cambios a GitHub.

## Comprobaciones

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
node --test tests/uploader.test.cjs
```

Las comprobaciones de Python cubren la integridad de los datos, métricas, compatibilidad de modelos guardados, validación de fotos subidas, estados con modelos ausentes o dañados, orden fijo y copias verificadas de modelos, explicaciones de los modelos, textos bilingües y conservación de predicciones al cambiar de idioma. Otras cuatro comprobaciones del componente de carga (Node.js, opcional para utilizar la aplicación) cubren eventos de carga, cambios de idioma, límites de archivos, eliminación de la selección y lecturas simultáneas. Las imágenes, los entornos, los pesos descargados y los modelos entrenados están excluidos de Git. Las mediciones globales del piloto y la correspondencia de destinos se publican en `experiments/pilot/`. Los manifests detallados de imágenes, inventarios de origen e informes por imagen de la preparación se mantienen locales porque pueden incluir identificadores privados de las fuentes.
