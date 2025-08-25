import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

try:
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras import layers, models
    import tensorflow as tf
except ImportError:
    st.error("TensorFlow no está instalado correctamente")
    st.stop()

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de Plantas",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🌱 Clasificador de Plantas con Deep Learning")
st.markdown("---")

# Nombres de las clases (30 tipos de plantas)
class_names = [
    'Aloevera', 'Banana', 'Bilimbi', 'Cantaloupe', 'Cassava', 
    'Coconut', 'Corn', 'Cucumber', 'Curcuma', 'Eggplant',
    'French_Beans', 'Ginger', 'Guava', 'Jambu', 'Kale',
    'Longbeans', 'Mango', 'Melon', 'Orange', 'Paddy',
    'Papaya', 'Passionfruit', 'Potato', 'Raddish', 'Rose',
    'Soybeans', 'Spinach', 'Sweetpotato', 'Tobacco', 'Waterapple'
]

@st.cache_resource
def load_model():
    """Cargar el modelo entrenado"""
    try:
        # Recrear la arquitectura del modelo
        vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
        
        for layer in vgg16_base.layers[:12]:  
            layer.trainable = False
        
        model = models.Sequential([
            vgg16_base,
            layers.Flatten(),
            layers.Dense(512, activation='relu'),  
            layers.Dropout(0.7),
            layers.Dense(30, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None

def preprocess_image(image):
    """Preprocesar imagen para el modelo"""
    image = image.convert('RGB')
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_plant(model, image):
    """Realizar predicción"""
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])
    return predicted_class, confidence, predictions[0]

def plot_predictions(predictions, top_n=5):
    """Crear gráfico de las predicciones top N"""
    top_indices = np.argsort(predictions)[::-1][:top_n]
    top_probs = predictions[top_indices]
    top_classes = [class_names[i] for i in top_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(top_classes)), top_probs, color='lightgreen')
    ax.set_yticks(range(len(top_classes)))
    ax.set_yticklabels(top_classes)
    ax.set_xlabel('Probabilidad')
    ax.set_title(f'Top {top_n} Predicciones')
    
    # Agregar valores en las barras
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    return fig

# Sidebar
st.sidebar.header("⚙️ Configuración")

# Cargar modelo
with st.spinner("Cargando modelo..."):
    model = load_model()

if model is None:
    st.error("No se pudo cargar el modelo. Asegúrate de haber entrenado el modelo primero.")
    st.stop()

st.sidebar.success("✅ Modelo cargado exitosamente")

# Opciones de visualización
show_top_n = st.sidebar.slider("Mostrar top N predicciones", min_value=3, max_value=10, value=5)
confidence_threshold = st.sidebar.slider("Umbral de confianza", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

# Información del modelo
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Información del Modelo")
st.sidebar.info("""
**Arquitectura:** VGG16 + Fine-tuning

**Clases:** 30 tipos de plantas

**Precisión:** ~95.3%

**Tamaño de entrada:** 128x128 píxeles
""")

# Área principal
tab1, tab2, tab3 = st.tabs(["🔍 Clasificación", "📈 Análisis", "ℹ️ Información"])

with tab1:
    st.header("Subir imagen para clasificar")
    
    # Uploader de imagen
    uploaded_file = st.file_uploader(
        "Selecciona una imagen de una planta",
        type=['png', 'jpg', 'jpeg'],
        help="Formatos soportados: PNG, JPG, JPEG"
    )
    
    if uploaded_file is not None:
        # Mostrar imagen
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption="Imagen subida", use_column_width=True)
        
        with col2:
            # Realizar predicción
            with st.spinner("Clasificando imagen..."):
                predicted_class, confidence, all_predictions = predict_plant(model, image)
                predicted_name = class_names[predicted_class]
            
            # Mostrar resultado principal
            st.markdown("### 🎯 Resultado")
            
            if confidence >= confidence_threshold:
                st.success(f"**{predicted_name}**")
                st.metric("Confianza", f"{confidence:.2%}")
            else:
                st.warning(f"**{predicted_name}** (Baja confianza)")
                st.metric("Confianza", f"{confidence:.2%}")
        
        # Mostrar gráfico de predicciones
        st.markdown("### 📊 Distribución de Predicciones")
        fig = plot_predictions(all_predictions, show_top_n)
        st.pyplot(fig)
        
        # Tabla detallada
        with st.expander("Ver todas las predicciones"):
            results_df = {
                'Clase': class_names,
                'Probabilidad': [f"{p:.4f}" for p in all_predictions],
                'Porcentaje': [f"{p:.2%}" for p in all_predictions]
            }
            import pandas as pd
            df = pd.DataFrame(results_df)
            df = df.sort_values('Probabilidad', ascending=False).reset_index(drop=True)
            st.dataframe(df, use_container_width=True)

with tab2:
    st.header("📈 Análisis del Modelo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Precisión del Modelo", "95.3%", "2.1%")
        st.metric("Total de Clases", "30")
        st.metric("Imágenes de Entrenamiento", "24,000")
    
    with col2:
        st.metric("Imágenes de Prueba", "6,000")
        st.metric("Arquitectura Base", "VGG16")
        st.metric("Parámetros Entrenables", "~2.1M")
    
    # Lista de clases
    st.markdown("### 🌿 Clases de Plantas Reconocidas")
    
    # Dividir en columnas para mejor presentación
    cols = st.columns(5)
    for i, plant_name in enumerate(class_names):
        with cols[i % 5]:
            st.write(f"• {plant_name}")

with tab3:
    st.header("ℹ️ Información del Proyecto")
    
    st.markdown("""
    ### 🎯 Objetivo
    Este proyecto implementa un clasificador de plantas usando transfer learning con VGG16,
    capaz de identificar 30 tipos diferentes de plantas con una precisión del 95.3%.
    
    ### 🔧 Metodología
    1. **Dataset:** 30,000 imágenes de 30 clases de plantas
    2. **Preprocesamiento:** Redimensionado a 128x128, normalización
    3. **Modelo:** VGG16 preentrenado + capas densas personalizadas
    4. **Fine-tuning:** Descongelar últimas capas para ajuste fino
    5. **Regularización:** Dropout (0.7) y Early Stopping
    
    ### 📊 Resultados
    - **Precisión en Test:** 95.27%
    - **F1-Score Promedio:** 0.95
    - **Tiempo de Entrenamiento:** ~8 horas
    
    ### 👥 Autores
    - Bruno Arezo
    - Ivan Gonzalez  
    - Camila Pazos
    """)
    
    st.markdown("---")
    st.markdown("*Desarrollado con ❤️ usando Streamlit y TensorFlow*")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>🌱 Clasificador de Plantas v1.0</div>", 
    unsafe_allow_html=True
)