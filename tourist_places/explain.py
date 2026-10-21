"""Pure, bilingual descriptions of a saved model and its schematic workflow.

The pilot identity below is the manifest digest recorded in
experiments/pilot/metadata.json. Counts must never be inferred from an
architecture or a run folder name. No checkpoint tensors or images are read here.
"""
from collections.abc import Mapping
import math


PILOT_MANIFEST_SHA256 = "65ab0d996ed647b427c324b9d2df9b3592385333cc3481d95fc9d7e1a81dc031"

_STYLES = """
    classDef stage fill:#E8F3EF,stroke:#14766D,color:#174D45,stroke-width:1.2px;
    classDef result fill:#D5EAE2,stroke:#14766D,color:#174D45,stroke-width:1.5px;
"""


def _nonnegative_int(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def explain_model(checkpoint, language="es"):
    """Return UI-ready text, Mermaid source, and accessible diagram alternatives.

    All fields are strings except ``is_pilot``. Missing metadata yields a generic
    explanation rather than invented architecture, training settings, or counts.
    The supplied mapping is never modified, and the function performs no I/O.
    """
    checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
    config = checkpoint.get("config")
    config = config if isinstance(config, Mapping) else {}
    classes = checkpoint.get("classes")
    count = len(classes) if isinstance(classes, (list, tuple)) else 0
    architecture = checkpoint.get("architecture")
    is_pilot = checkpoint.get("manifest_sha256") == PILOT_MANIFEST_SHA256 and count == 164
    english = language == "en"

    def tr(spanish, english_text):
        return english_text if english else spanish

    destination_label = tr(f"{count} destinos guardados", f"{count} saved destinations") if count else tr("Destinos guardados", "Saved destinations")
    top_label = tr(f"Hasta {min(5, count)} opciones", f"Up to {min(5, count)} candidates") if count else tr("Mejores opciones", "Top candidates")
    upload = tr("Tu foto", "Your photo")
    softmax = tr("Softmax: puntuaciones relativas", "Softmax: relative scores")
    ranked = top_label + "<br/>" + tr("Comparar con el umbral de incertidumbre", "Compare with the uncertainty threshold")

    if architecture == "efficientnet_b0":
        preprocess = tr("Orientación EXIF y RGB<br/>Reescalar a 256 y recortar a 224 × 224<br/>Normalizar con ImageNet", "EXIF orientation and RGB<br/>Resize to 256 and crop to 224 × 224<br/>Normalize with ImageNet")
        features = tr("EfficientNet-B0<br/>Extraer rasgos visuales", "EfficientNet-B0<br/>Extract visual features")
        classifier = tr("Promedio global + capa lineal", "Global average pooling + linear layer") + "<br/>" + destination_label
        build_text = tr(
            "EfficientNet-B0 extrae patrones visuales, los resume con un promedio global y utiliza una capa lineal para puntuar cada destino guardado. Su clasificador incluye Dropout, activo solo durante el entrenamiento. Para predecir, se corrige la orientación EXIF, se convierte a RGB, se reescala el lado corto a 256 píxeles, se recorta el centro a 224 × 224 y se aplica la normalización de ImageNet.",
            "EfficientNet-B0 extracts visual patterns, summarizes them with global average pooling, and uses a linear layer to score each saved destination. Its classifier includes Dropout, active only during training. Prediction corrects EXIF orientation, converts to RGB, resizes the short side to 256 pixels, center-crops to 224 × 224, and applies ImageNet normalization.")
        if config.get("no_pretrained") is False:
            build_text += tr(" Esta ejecución parte de pesos aprendidos en ImageNet y sustituye la última capa por el clasificador de destinos.", " This run starts with weights learned on ImageNet and replaces the final layer with the destination classifier.")
        elif config.get("no_pretrained") is True:
            build_text += tr(" Esta ejecución desactivó los pesos preentrenados: aprendió desde cero y no representa la referencia de transferencia.", " This run disabled pretrained weights: it learned from scratch and does not represent the transfer-learning baseline.")
        else:
            build_text += tr(" Este archivo no registra si se utilizaron pesos preentrenados al iniciar.", " This file does not record whether pretrained weights were used initially.")
        inference_alt = tr(
            f"Foto: corregir EXIF y convertir a RGB; reescalar y recortar a 224 por 224; normalizar con ImageNet; extraer rasgos con EfficientNet-B0; promedio global y clasificador de {count} destinos; softmax; ordenar hasta {min(5, count)} opciones y comparar la primera puntuación con el umbral.",
            f"Photo: correct EXIF and convert to RGB; resize and crop to 224 by 224; normalize with ImageNet; extract features with EfficientNet-B0; global pooling and a {count}-destination classifier; softmax; rank up to {min(5, count)} candidates and compare the first score with the threshold.")
    elif architecture == "legacy_cnn":
        preprocess = tr("Orientación EXIF y RGB<br/>Reescalar a 146 y recortar a 128 × 128<br/>Tensor sin normalización de ImageNet", "EXIF orientation and RGB<br/>Resize to 146 and crop to 128 × 128<br/>Tensor without ImageNet normalization")
        features = tr("3 etapas: Conv + ReLU + MaxPool<br/>16 → 32 → 64 canales", "3 stages: Conv + ReLU + MaxPool<br/>16 → 32 → 64 channels")
        classifier = tr("Aplanar → densa de 512 + ReLU<br/>Capa lineal: ", "Flatten → 512-unit dense + ReLU<br/>Linear layer: ") + destination_label
        build_text = tr(
            "La CNN original adaptada aprende desde cero. Tiene tres etapas de convolución, ReLU y MaxPool con 16, 32 y 64 canales. Después aplana los rasgos y utiliza una capa densa de 512 unidades con ReLU y una salida lineal por destino. Sustituye las salidas jerárquicas del cuaderno original. Para predecir, corrige EXIF, convierte a RGB, reescala el lado corto a 146 píxeles y recorta el centro a 128 × 128. Convierte los píxeles a un tensor entre 0 y 1, sin normalización de ImageNet.",
            "The adapted original CNN learns from scratch. It has three convolution, ReLU, and MaxPool stages with 16, 32, and 64 channels. It then flattens features and uses a 512-unit dense layer with ReLU and one linear output per destination. It replaces the original notebook's hierarchical outputs. Prediction corrects EXIF, converts to RGB, resizes the short side to 146 pixels, and center-crops to 128 × 128. Pixels become a tensor between 0 and 1, without ImageNet normalization.")
        inference_alt = tr(
            f"Foto: corregir EXIF y convertir a RGB; reescalar y recortar a 128 por 128, sin normalización de ImageNet; tres etapas Conv, ReLU y MaxPool de 16, 32 y 64 canales; aplanado, capa densa de 512 y salida para {count} destinos; softmax; ordenar hasta {min(5, count)} opciones y aplicar el umbral.",
            f"Photo: correct EXIF and convert to RGB; resize and crop to 128 by 128 without ImageNet normalization; three Conv, ReLU, and MaxPool stages of 16, 32, and 64 channels; flatten, 512-unit dense layer, and {count} destination outputs; softmax; rank up to {min(5, count)} candidates and apply the threshold.")
    else:
        preprocess = tr("Validar y preparar la imagen<br/>Según el modelo guardado", "Validate and prepare the image<br/>According to the saved model")
        features = tr("Red del modelo seleccionado", "Selected model network")
        classifier = tr("Clasificador de destinos", "Destination classifier") + "<br/>" + destination_label
        build_text = tr(
            "Este archivo no identifica una arquitectura compatible para describir sus capas o el tamaño de entrada. El esquema muestra únicamente el flujo general de reconocimiento, sin atribuirle una red preentrenada ni ajustes no registrados.",
            "This file does not identify a supported architecture whose layers or input size can be described. The diagram shows only the general recognition flow, without assuming a pretrained network or unrecorded settings.")
        inference_alt = tr(
            "Una foto se valida y prepara según el modelo guardado, pasa por su red y clasificador de destinos, y produce puntuaciones relativas. Se ordenan las mejores opciones y se compara la primera con el umbral de incertidumbre; no se conocen las capas ni el tamaño de entrada.",
            "A photo is validated and prepared according to the saved model, passes through its network and destination classifier, and produces relative scores. The top candidates are ranked and the first is compared with the uncertainty threshold; the layers and input size are unknown.")

    inference_mermaid = f'''flowchart TD
    A["{upload}"] --> B["{preprocess}"]
    B --> C["{features}"]
    C --> D["{classifier}"]
    D --> E["{softmax}"]
    E --> F["{ranked}"]
{_STYLES}
    class A,B,C,D,E stage;
    class F result;
'''

    training_parts = []
    warmup = _nonnegative_int(config.get("warmup_epochs"))
    epochs = _nonnegative_int(config.get("epochs"))
    if architecture == "legacy_cnn":
        training_parts.append(tr("Se entrenan desde cero todas las capas de la CNN y el clasificador.", "All CNN layers and the classifier are trained from scratch."))
    elif architecture == "efficientnet_b0":
        if warmup is not None and epochs and warmup >= epochs:
            training_parts.append(tr("Solo aprende el clasificador: el extractor de características y sus estadísticas de BatchNorm permanecen congelados durante toda la ejecución.", "Only the classifier learns: the feature extractor and its BatchNorm statistics stay frozen throughout the run."))
        elif warmup is not None:
            if warmup:
                training_parts.append(tr(f"La configuración comienza con {warmup} épocas de extractor congelado y después permite ajustar sus pesos junto con el clasificador.", f"The configuration starts with {warmup} epochs of a frozen feature extractor, then allows its weights to adapt along with the classifier."))
            else:
                training_parts.append(tr("La configuración permite entrenar los pesos del extractor y del clasificador desde la primera época.", "The configuration allows the feature extractor and classifier weights to train from the first epoch."))
            if config.get("update_batchnorm") is False:
                training_parts.append(tr("Durante el ajuste fino se conservan fijas las estadísticas acumuladas de BatchNorm.", "BatchNorm running statistics stay fixed during fine-tuning."))
            elif config.get("update_batchnorm") is True:
                training_parts.append(tr("Durante el ajuste fino también se actualizan las estadísticas acumuladas de BatchNorm.", "Fine-tuning also updates BatchNorm running statistics."))
            else:
                training_parts.append(tr("El archivo no registra la política de actualización de BatchNorm; no se puede confirmar ese ajuste.", "The file does not record its BatchNorm update policy; that setting cannot be confirmed."))
        else:
            training_parts.append(tr("El archivo no registra la duración de la fase congelada ni permite reconstruir la estrategia de ajuste fino.", "The file does not record the frozen-phase duration or allow its fine-tuning strategy to be reconstructed."))
    else:
        training_parts.append(tr("Faltan los datos de arquitectura necesarios para describir cómo se entrenaron sus capas.", "The architecture metadata needed to describe how its layers were trained is missing."))

    if epochs:
        training_parts.append(tr(f"El límite configurado es de {epochs} épocas; la parada temprana puede terminar antes.", f"The configured limit is {epochs} epochs; early stopping may finish sooner."))
    batch = _nonnegative_int(config.get("batch_size"))
    if batch:
        training_parts.append(tr(f"Tamaño de lote: {batch} fotos.", f"Batch size: {batch} photos."))
    training_parts.append(tr(
        "En este flujo, solo entrenamiento ajusta los pesos y recibe aumentos aleatorios de imagen. La validación selecciona la mejor época por macro-F1, con la pérdida como desempate. La prueba reservada se utiliza al final y no alimenta el entrenamiento ni esa selección.",
        "In this workflow, only training updates weights and receives random image augmentation. Validation selects the best epoch by macro-F1, with loss breaking ties. Held-out test data is used at the end and does not feed training or that selection."))

    if is_pilot:
        source = tr("4 668 fotos de la copia local", "4,668 photos from the local copy")
        clean = tr("Agrupar duplicados y reflejos<br/>Apartar 120 fotos con etiquetas en conflicto", "Group duplicates and mirrored photos<br/>Quarantine 120 conflicting-label photos")
        split = tr("Filtrar destinos con pocas fotos<br/>1 375 fotos independientes · 164 destinos", "Filter destinations with too few photos<br/>1,375 independent photos · 164 destinations")
        train = tr("957 de entrenamiento<br/>Aprender pesos", "957 training photos<br/>Learn weights")
        validation = tr("209 de validación<br/>Comparar épocas", "209 validation photos<br/>Compare epochs")
        test = tr("209 de prueba<br/>Reservadas hasta el final", "209 test photos<br/>Held out until the end")
        training_alt = tr(
            "De 4 668 fotos locales se agrupan duplicados y reflejos, se apartan 120 fotos con etiquetas contradictorias y se filtran destinos con pocas fotos. Quedan 1 375 fotos independientes de 164 destinos: 957 para aprender pesos y 209 para seleccionar la mejor época por macro-F1 de validación. Las otras 209 fotos se reservan para la evaluación final del modelo seleccionado y no alimentan el aprendizaje.",
            "From 4,668 local photos, duplicates and mirrors are grouped, 120 conflicting-label photos are quarantined, and destinations with too few photos are filtered. This leaves 1,375 independent photos of 164 destinations: 957 learn weights and 209 select the best epoch by validation macro-F1. The other 209 photos are held out for final evaluation of the selected model and do not feed learning.")
        training_parts.append(tr("Las cantidades corresponden al piloto verificado por la identidad exacta de su división; el modelo no cubre todos los destinos originales.", "These counts belong to the pilot verified by its exact split identity; the model does not cover every original destination."))
    else:
        source = tr("Fotos de origen", "Source photos")
        clean = tr("Revisar etiquetas<br/>Agrupar duplicados", "Review labels<br/>Group duplicates")
        split = tr("Dividir en conjuntos separados", "Create separate data splits")
        train = tr("Entrenamiento<br/>Aprender pesos", "Training<br/>Learn weights")
        validation = tr("Validación<br/>Comparar épocas", "Validation<br/>Compare epochs")
        test = tr("Prueba<br/>Reservada hasta el final", "Test<br/>Held out until the end")
        training_alt = tr(
            "Esquema general: revisar etiquetas y agrupar duplicados antes de crear entrenamiento, validación y prueba separados. Entrenamiento aprende los pesos; validación selecciona la mejor época por macro-F1; prueba se reserva para la evaluación final y no alimenta el aprendizaje. No se muestran cantidades porque la división del piloto no está verificada para este archivo.",
            "General workflow: review labels and group duplicates before creating separate training, validation, and test splits. Training learns weights; validation selects the best epoch by macro-F1; test is held out for final evaluation and does not feed learning. Counts are omitted because the pilot split is not verified for this file.")
        training_parts.append(tr("Se muestra un esquema general: no se ha verificado que este archivo utilice la división del piloto, por lo que no se le atribuyen sus cantidades.", "This is a general diagram: this file has not been verified as using the pilot split, so its counts are not attributed to it."))

    saved = tr("Guardar mejor época<br/>Macro-F1 de validación", "Save the best epoch<br/>Validation macro-F1")
    final = tr("Evaluación final<br/>Sin actualizar los pesos", "Final evaluation<br/>No weight updates")
    training_mermaid = f'''flowchart TD
    A["{source}"] --> B["{clean}"]
    B --> C["{split}"]
    C --> T["{train}"]
    C --> V["{validation}"]
    C --> H["{test}"]
    T --> S["{saved}"]
    V --> S
    S --> E["{final}"]
    H --> E
{_STYLES}
    class A,B,C,T,V,H stage;
    class S,E result;
'''
    epoch_text = ""
    epoch = _nonnegative_int(checkpoint.get("epoch"))
    if epoch:
        epoch_text = tr(f"Época guardada: {epoch}.", f"Saved epoch: {epoch}.")
        metrics = checkpoint.get("val_metrics")
        score = metrics.get("macro_f1") if isinstance(metrics, Mapping) else None
        if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score):
            number = f"{score:.3f}"
            if not english:
                number = number.replace(".", ",")
            epoch_text += tr(f" Macro-F1 de validación: {number}.", f" Validation macro-F1: {number}.")

    return {
        "build_title": tr("Cómo se construyó este modelo", "How this model was built"),
        "build_text": build_text,
        "inference_title": tr("De tu foto a los destinos sugeridos", "From your photo to suggested destinations"),
        "inference_mermaid": inference_mermaid,
        "inference_alt": inference_alt,
        "training_title": tr("Del conjunto de fotos al modelo elegido", "From the photo collection to the selected model"),
        "training_mermaid": training_mermaid,
        "training_alt": training_alt,
        "training_text": " ".join(training_parts),
        "epoch_text": epoch_text,
        "schematic_note": tr("Los diagramas resumen las etapas del proceso. No son mapas de atención ni muestran qué zona de una foto causó una predicción.", "The diagrams summarize process stages. They are not attention maps and do not show which part of a photo caused a prediction."),
        "score_note": tr("Softmax compara los destinos guardados; su puntuación no es una probabilidad calibrada de acierto. Un lugar desconocido también puede recibir una puntuación alta. El umbral solo marca resultados como inciertos.", "Softmax compares saved destinations; its score is not a calibrated probability of being correct. An unknown place can also receive a high score. The threshold only marks results as uncertain."),
        "is_pilot": is_pilot,
    }
