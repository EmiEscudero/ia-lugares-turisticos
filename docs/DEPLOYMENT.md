# Publish on Streamlit Community Cloud

The public app uses four sanitized trained model files hosted as versioned GitHub release assets. Their expected sizes and SHA-256 digests are pinned in `cloud/models.json`. Source photographs, detailed image manifests, local paths, and private collection links are not needed by the hosted app.

## Exact deployment settings

Sign in yourself at [Streamlit Community Cloud](https://share.streamlit.io/). The sign-in page requires agreement to Streamlit's terms; complete that account step yourself.

Choose **Create app**, then **Yup, I have an app**, and use:

| Field | Value |
| --- | --- |
| Repository | `EmiEscudero/ia-lugares-turisticos` |
| Branch | `codex/improve-place-recognition` |
| Main file path | `cloud/streamlit_app.py` |
| Advanced settings → Python version | `3.13` |
| Secrets | None required |

Choose an available app subdomain if desired, then **Deploy**. The entrypoint's `cloud/requirements.txt` installs CPU-only PyTorch and Streamlit; it takes precedence over the root training requirements. First startup downloads approximately 86 MB of model assets and verifies their digests. Four numbered models are then available. CPU inference is limited to two threads, and at most two loaded models are cached.

Copy the final `https://…streamlit.app` address from the successful deployment. Use `?lang=es` or `?lang=en` for portfolio links. A GitHub push or a model release alone does not create that live URL. The `main` branch does not need to change.

## Privacy and local use

The hosted app receives uploaded photos on its server and keeps them in memory; the application does not save uploads. Public model assets retain model weights, destination labels, evaluation metrics, and scalar training settings. They exclude source image records and local source/output folder paths. No private photo collection access is required.

For a preview accessible only on your computer, run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

The shared repository configuration leaves server binding to the hosting environment. The explicit local argument above restricts your local preview to loopback.

Removing private links from the current branch does not change the original storage permissions and does not erase earlier Git commits. Detailed image manifests and preparation reports remain available only in the local working copy; they are excluded from the current public snapshot.

## If deployment fails

- Check the repository, development branch, entrypoint, and Python version above. Deploying root `app.py` directly will look for local training runs rather than setting up hosted models.
- If package installation tries to install CUDA libraries, verify that `cloud/requirements.txt` is the selected entrypoint's requirements file.
- If the model download fails, confirm that the `streamlit-pilot-v1` release and its four assets are public, then retry or reboot the app. The app rejects incomplete or altered files instead of loading them.
- Read the Community Cloud build/runtime logs for an installation or startup error. No Drive credentials or links should be added as a workaround.

See the official [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [file organization](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization), and [dependency rules](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

## Instrucciones en español

Inicia sesión tú mismo en [Streamlit Community Cloud](https://share.streamlit.io/) y completa la aceptación de sus términos. Selecciona **Create app** y **Yup, I have an app**. Usa el repositorio `EmiEscudero/ia-lugares-turisticos`, la rama `codex/improve-place-recognition` y el archivo `cloud/streamlit_app.py`. En **Advanced settings**, elige Python `3.13`. No necesitas secretos ni acceso a la colección de fotos. Elige un subdominio disponible y pulsa **Deploy**.

El primer arranque descarga y verifica los cuatro modelos publicados en GitHub; puede tardar un poco. Cuando termine, copia la dirección `https://…streamlit.app` y agrega `?lang=es` o `?lang=en` para tu portafolio. Publicar el código en GitHub no crea por sí solo esa dirección y no es necesario cambiar `main`.

Las fotos subidas se envían al servidor para analizarlas y la aplicación no las guarda. Los modelos públicos no contienen rutas locales ni registros de las fotos de origen. Los enlaces privados se retiraron de la versión actual; esto no cambia los permisos del almacenamiento original ni borra los commits anteriores. Para ejecutar una vista previa solo en tu equipo, utiliza el comando local anterior con `--server.address 127.0.0.1`.
