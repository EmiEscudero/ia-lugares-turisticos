"""Interface copy. Destination and state names remain geographic proper names."""

from .model_catalog import MODEL_ORDER, model_run_name

LANGUAGES = {"es": "Español", "en": "English"}

# Keep each message together so changes cannot silently omit one language.
MESSAGES = {
    "page_title": ("México · Reconoce lugares", "México · Place Finder"),
    "brand": ("RECONOCE LUGARES", "PLACE FINDER"),
    "model": ("Modelo de reconocimiento", "Recognition model"),
    "model_picker_title": ("Elige el modelo", "Choose a model"),
    "model_order_note": ("Orden fijo según los resultados de validación: recomendado primero. Los números identifican las opciones; no son porcentajes.", "Fixed order based on validation results: recommended first. The numbers identify the options; they are not percentages."),
    "model_comparison": ("Los cuatro modelos, en orden", "The four models, in order"),
    "model_comparison_note": ("El recomendado se eligió con fotos de validación antes de evaluar las fotos de prueba. Los resultados de prueba de esta tabla no determinan el orden.", "The recommended model was chosen using validation photos before evaluating test photos. The test results in this table do not determine the order."),
    "model_column": ("Modelo", "Model"),
    "approach_column": ("Qué cambia", "What changes"),
    "top5_column": ("Entre las 5 opciones", "In the top 5"),
    "not_evaluated": ("Sin evaluación", "Not evaluated"),
    "additional_models": ("Los modelos adicionales aparecen después, por nombre.", "Additional models appear afterward, by name."),
    "quality_selected": ("Resultados del modelo seleccionado", "Selected model results"),
    "training_details": ("Detalles del entrenamiento y de la selección", "Training and model selection details"),
    "prediction_flow": ("Cómo predice", "How it predicts"),
    "training_flow": ("Cómo se entrenó y evaluó", "How it was trained and evaluated"),
    "threshold": ("Puntuación mínima de coincidencia", "Minimum match score"),
    "threshold_help": ("Auméntala para marcar más resultados como inciertos. La puntuación no garantiza un acierto.", "Raise this to show more results as uncertain. A match score is not a guarantee."),
    "refresh": ("Actualizar modelos disponibles", "Refresh available models"),
    "research": ("Un proyecto en investigación", "A research preview"),
    "research_note": ("Funciona mejor con puntos de referencia y vistas claras. Algunas playas se parecen mucho; un lugar fuera de la colección también puede recibir una coincidencia.", "Best with recognizable landmarks and clear views. Some beaches look very similar, and places outside the collection may still receive a match."),
    "privacy": ("Tu foto se procesa en este equipo y la aplicación no la guarda.", "Your photo is processed on this computer and is not saved by this app."),
    "hosted_privacy": ("Tu foto se envía al servidor de esta aplicación para analizarla. La aplicación la mantiene en memoria y no la guarda.", "Your photo is sent to this app's server for analysis. The app keeps it in memory and does not save it."),
    "hosted_loading": ("Preparando los cuatro modelos publicados… La primera visita puede tardar un poco.", "Preparing the four published models… The first visit may take a moment."),
    "hosted_error": ("No se pudieron preparar los modelos publicados. Inténtalo de nuevo en unos momentos.", "The published models could not be prepared. Please try again in a moment."),
    "retry_models": ("Intentar de nuevo", "Try again"),
    "hosted_footer": ("Un proyecto de investigación para viajeros curiosos.", "A research project for curious travelers."),
    "eyebrow": ("EXPLORA MÉXICO, UNA FOTO A LA VEZ", "EXPLORE MEXICO, ONE PHOTO AT A TIME"),
    "hero": ("¿Dónde podría ser?", "Where could this be?"),
    "intro": ("Una playa, una plaza, un lugar que recuerdas. Agrega una foto y descubre los destinos más parecidos de nuestra colección.", "A beach, a plaza, a place you remember. Add a photo to discover the closest destinations in our collection."),
    "loading": ("Preparando el modelo de reconocimiento…", "Preparing the recognition model…"),
    "model_error": ("No se pudo abrir este modelo. Elige otro o actualiza cuando termine el entrenamiento.", "This saved model could not be opened. Choose another model or refresh after training finishes."),
    "model_pending": ("Estamos preparando el modelo. La página estará lista cuando haya un modelo entrenado; usa «Actualizar modelos disponibles» para comprobarlo.", "The recognition model is being prepared. This page will be ready when a trained model is available; use ‘Refresh available models’ to check again."),
    "check_model": ("Buscar un modelo entrenado", "Check for a trained model"),
    "explore_tab": ("Explorar una foto", "Explore a photo"),
    "quality_tab": ("Acerca del modelo", "About the model"),
    "start": ("Comienza con una foto", "Start with a photo"),
    "choose_photo": ("Elegir una foto", "Choose a photo"),
    "drop_photo": ("Arrastra tu foto aquí", "Drop your photo here"),
    "upload_help": ("JPEG, PNG o WebP · hasta 20 MB y 30 megapíxeles", "JPEG, PNG, or WebP · up to 20 MB and 30 megapixels"),
    "remove_photo": ("Quitar foto", "Remove photo"),
    "uploading": ("Leyendo la foto…", "Reading the photo…"),
    "example_label": ("O prueba un ejemplo", "Or try an example"),
    "choose_example": ("Elige un ejemplo", "Choose an example"),
    "example_caption": ("ejemplo de la colección de validación", "example from the validation collection"),
    "analyze": ("Encontrar este lugar", "Find this place"),
    "analyzing": ("Buscando lugares conocidos…", "Looking for familiar places…"),
    "inference_error": ("No se pudo analizar la foto. Prueba con otra imagen o modelo.", "The photo could not be analyzed. Try another image or model."),
    "destinations": ("Tus posibles destinos", "Your possible destinations"),
    "uncertain": ("No hay una coincidencia suficientemente fuerte para identificar el lugar. Estas son posibilidades para explorar.", "There isn't a strong enough match to identify this place. These are possibilities to explore."),
    "candidate": ("Se encontró una posible coincidencia. Compara las alternativas antes de decidir.", "A possible match was found. Compare the alternatives before deciding."),
    "closest_score": ("Puntuación de la mejor coincidencia", "Closest match score"),
    "score_note": ("Las puntuaciones comparan destinos conocidos; no son probabilidades de que el lugar sea correcto. Un lugar desconocido también puede recibir una puntuación alta.", "Scores compare known destinations; they are not probabilities that a place is correct. An unfamiliar place can still receive a high score."),
    "download": ("Guardar estos resultados", "Save these results"),
    "download_name": ("coincidencias_destinos.json", "destination_matches.json"),
    "empty_title": ("Un poco de investigación visual", "A little visual detective work"),
    "empty_copy": ("Aquí aparecerán las mejores coincidencias, con el estado y una puntuación para cada destino.", "Your closest matches will appear here, along with the state and a score for each destination."),
    "photo_tip": ("Prueba con un edificio, una costa o un punto de referencia distintivo. Las fotos amplias y claras aportan más contexto.", "Try a distinctive building, coastline, or landmark. Wide, clear photos give the model more context."),
    "val_photos": ("Fotos de validación", "Validation photos"),
    "test_photos": ("Fotos de prueba reservadas", "Held-out test photos"),
    "quality_title": ("¿Qué tan bien reconoce los lugares?", "How well does this model recognize places?"),
    "quality_note": ("Estas cifras provienen de fotos etiquetadas, no de la foto que subes. Los resultados de investigación pueden diferir de los de fotos de viaje cotidianas.", "These numbers come from labeled photos, not from the photo you upload. Research results can differ from everyday travel photos."),
    "coverage": ("Destinos incluidos", "Destinations included"),
    "top1": ("Primera opción correcta", "Correct first choice"),
    "recall": ("Proporción media de aciertos por destino: {score}. Las escenas parecidas y los lugares con pocas fotos siguen siendo difíciles.", "Average recall across destinations: {score}. Similar scenes and underrepresented places remain challenging."),
    "collection": ("Destinos de esta colección", "Destinations in this collection"),
    "destination_column": ("Destino", "Destination"),
    "state_column": ("Estado", "State"),
    "type_column": ("Tipo", "Type"),
    "beach": ("playa", "beach"),
    "town": ("pueblo", "town"),
    "quality_pending": ("La cobertura y los resultados del modelo aparecerán cuando termine el entrenamiento.", "Model coverage and evaluation results will appear here once training is complete."),
    "footer": ("Un proyecto local de investigación para viajeros curiosos.", "A local research project for curious travelers."),
    "upload_size": ("Elige una foto de hasta 20 MB.", "Choose a photo no larger than 20 MB."),
    "upload_format": ("Elige una foto JPEG, PNG o WebP.", "Choose a JPEG, PNG, or WebP photo."),
    "upload_pixels": ("Elige una foto de hasta 30 millones de píxeles.", "Choose a photo with no more than 30 million pixels."),
    "upload_unreadable": ("No se pudo leer este archivo como foto. Prueba con otra imagen.", "This file could not be read as a photo. Try another image."),
    "recommended": ("Recomendado", "Recommended"),
    "frozen_pilot": ("Características fijas", "Frozen features"),
    "stable_transfer_pilot": ("Ajuste fino estable · Recomendado", "Stable fine-tuning · Recommended"),
    "transfer_pilot": ("Ajuste fino inicial", "Initial fine-tuning"),
    "legacy_pilot": ("CNN original · Referencia", "Original CNN · Baseline"),
    "frozen_pilot_description": ("EfficientNet-B0 reutiliza sus características preentrenadas y aprende solo el clasificador de destinos.", "EfficientNet-B0 reuses its pretrained features and learns only the destination classifier."),
    "stable_transfer_pilot_description": ("EfficientNet-B0 adapta sus características a los destinos y conserva las estadísticas preentrenadas. Es la opción recomendada para comenzar.", "EfficientNet-B0 adapts its features to the destinations while preserving pretrained statistics. This is the recommended starting point."),
    "transfer_pilot_description": ("EfficientNet-B0 adapta características y estadísticas. Se conserva para comparar el primer experimento de ajuste fino.", "EfficientNet-B0 adapts features and statistics. It is kept to compare the initial fine-tuning experiment."),
    "legacy_pilot_description": ("La CNN del diseño original aprende desde cero. Sirve como referencia para medir el aporte del aprendizaje por transferencia.", "The original CNN design learns from scratch. It provides a baseline for measuring the benefit of transfer learning."),
    "recommended_description": ("Copia local del modelo seleccionado para usar en la aplicación.", "Local copy of the model selected for the application."),
    "other_model_description": ("Modelo adicional guardado en este equipo.", "Additional model saved on this computer."),
    "stable_transfer_pilot_approach": ("Ajuste fino; estadísticas fijas", "Fine-tuning; fixed statistics"),
    "frozen_pilot_approach": ("Solo aprende el clasificador", "Learns only the classifier"),
    "transfer_pilot_approach": ("Ajusta características y estadísticas", "Adapts features and statistics"),
    "legacy_pilot_approach": ("Aprende desde cero", "Learns from scratch"),
}


def text(key, language="es", **values):
    return MESSAGES[key][1 if language == "en" else 0].format(**values)


def percentage(value, language="es"):
    number = f"{value * 100:.1f}"
    return number.replace(".", ",") + " %" if language == "es" else number + "%"


def model_name(path, language):
    name = model_run_name(path)
    label = text(name, language) if name in MESSAGES else name.replace("_", " ").title()
    return f"{MODEL_ORDER.index(name) + 1}. {label}" if name in MODEL_ORDER else label


def model_description(path, language):
    key = model_run_name(path) + "_description"
    return text(key if key in MESSAGES else "other_model_description", language)
