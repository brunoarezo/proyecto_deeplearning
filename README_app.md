# 🌿 Clasificador de Plantas VGG16

Interfaz web interactiva para clasificar plantas usando un modelo VGG16 fine-tuned.

## Instalación

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Ejecutar la aplicación:
```bash
streamlit run app_plantas.py
```

## Características

- **Clasificación de 30 especies de plantas**
- **Interfaz amigable** con Streamlit
- **Visualización de resultados** con gráficos
- **Top 5 predicciones** con probabilidades
- **Modelo VGG16** con precisión del ~95%

## Uso

1. Sube una imagen de planta en formato JPG, PNG o JPEG
2. Haz clic en "Clasificar Planta"
3. Observa los resultados con probabilidades y gráficos

## Especies Soportadas

El modelo puede clasificar entre 30 especies diferentes de plantas, incluyendo:
- Aloe vera
- Monstera deliciosa
- Ficus benjamina
- Y 27 especies más...

## Consejos para mejores resultados

- Usa imágenes claras y bien iluminadas
- Asegúrate de que la planta sea el objeto principal
- Evita fondos complejos
- Imágenes mínimo 128x128 píxeles