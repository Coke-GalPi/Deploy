# Proyecto de Data Mining

Este proyecto implementa una aplicación web utilizando Flask para predecir resultados basados en modelos de Árbol de Decisión y Bosque Aleatorio entrenados y guardados en formato ONNX.

## Instalación

1. Clona el repositorio:
	```sh
	git clone <URL_DEL_REPOSITORIO>
	cd <NOMBRE_DEL_REPOSITORIO>
	```

2. Crea y activa un entorno virtual:
	```sh
	python -m venv env
	source env/Scripts/activate  # En Windows
	source env/bin/activate      # En Unix o MacOS
	```

3. Instala las dependencias:
	```sh
	pip install -r requirements.txt
	```

## Uso

1. Ejecuta la aplicación Flask:
	```sh
	python main.py
	```

2. Abre tu navegador web y navega a `http://localhost:5000`.

## Rutas Disponibles

- `/` o `/index`: Página principal.
- `/randomForest`: Página para predicciones usando el modelo de Bosque Aleatorio.
- `/decisionTree`: Página para predicciones usando el modelo de Árbol de Decisión.
- `/predictRF`: Endpoint para realizar predicciones con el modelo de Bosque Aleatorio (método POST).
- `/predictDT`: Endpoint para realizar predicciones con el modelo de Árbol de Decisión (método POST).

## Estructura de los Archivos

- `app.py`: Archivo principal de la aplicación Flask.
- `main.py`: Archivo que carga los modelos y define las rutas.
- `data/`: Carpeta que contiene los modelos ONNX.
- `templates/`: Carpeta que contiene las plantillas HTML.

## Requisitos

Las dependencias del proyecto están listadas en el archivo [requirements.txt](requirements.txt).
