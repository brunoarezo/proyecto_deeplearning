import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import pandas as pd
import pickle

try:
    import tensorflow as tf
except ImportError:
    st.error("TensorFlow no está instalado correctamente")
    st.stop()

# Configuración de la página
st.set_page_config(
    page_title="🌱 PlantAI Classifier",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar el diseño
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #4CAF50, #2E7D32);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
    }
    .prediction-result {
        background: linear-gradient(135deg, #E8F5E8, #C8E6C9);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #4CAF50;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Título principal con estilo
st.markdown('<h1 class="main-header">🌱 PlantAI Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Clasificación inteligente de plantas con Deep Learning</p>', unsafe_allow_html=True)
st.markdown("---")


@st.cache_resource
def load_model_and_classes():
    """Cargar el modelo y nombres de clases"""
    try:
        # Cargar modelo
        model = tf.keras.models.load_model('modelo_plantas.keras')
        
        # Cargar nombres de clases
        with open('class_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
        
        return model, class_names
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None, None


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


def create_predictions_chart(predictions, class_names, top_n=5):
    """Crear gráfico interactivo de las predicciones top N con Plotly"""
    top_indices = np.argsort(predictions)[::-1][:top_n]
    top_probs = predictions[top_indices]
    top_classes = [class_names[i] for i in top_indices]
    
    # Colores degradados del verde
    colors = ['#1B5E20', '#2E7D32', '#388E3C', '#43A047', '#4CAF50'][:len(top_classes)]
    
    fig = go.Figure(data=[
        go.Bar(
            x=top_probs,
            y=top_classes,
            orientation='h',
            text=[f'{prob:.1%}' for prob in top_probs],
            textposition='inside',
            textfont=dict(color='white', size=12, family='Arial Black'),
            marker=dict(
                color=colors,
                line=dict(color='rgba(50,50,50,0.8)', width=1)
            ),
            hovertemplate='<b>%{y}</b><br>Confianza: %{x:.1%}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(
            text=f'🎯 Top {top_n} Predicciones',
            x=0.5,
            font=dict(size=20, color='#2E7D32')
        ),
        xaxis_title='Nivel de Confianza',
        yaxis_title='',
        height=300 + (top_n * 30),
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Arial', size=11),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            tickformat='.0%'
        ),
        yaxis=dict(
            showgrid=False,
            autorange='reversed'
        )
    )
    
    return fig


def create_confidence_chart(confidence):
    """Crear un gráfico de confianza elegante"""
    confidence_percent = confidence * 100
    
    # Determinar color y mensaje según el nivel de confianza
    if confidence_percent >= 80:
        color = '#4CAF50'  # Verde
        color_bg = '#E8F5E8'
        status = 'Excelente'
        icon = '🎯'
    elif confidence_percent >= 60:
        color = '#FF9800'  # Naranja
        color_bg = '#FFF3E0' 
        status = 'Buena'
        icon = '👍'
    else:
        color = '#F44336'  # Rojo
        color_bg = '#FFEBEE'
        status = 'Baja'
        icon = '⚠️'
    
    # Crear gráfico de barras circular (donut)
    fig = go.Figure(data=[
        go.Pie(
            values=[confidence_percent, 100 - confidence_percent],
            labels=['Confianza', ''],
            hole=0.6,
            marker=dict(
                colors=[color, '#E0E0E0'],
                line=dict(color='white', width=2)
            ),
            textinfo='none',
            showlegend=False,
            hovertemplate='<b>Confianza</b><br>%{value:.1f}%<extra></extra>'
        )
    ])
    
    # Añadir texto central
    fig.add_annotation(
        text=f"<b>{confidence_percent:.1f}%</b><br><span style='font-size:14px; color:{color};'>{icon} {status}</span>",
        x=0.5, y=0.5,
        font=dict(size=24, color=color, family='Arial Black'),
        showarrow=False
    )
    
    fig.update_layout(
        title=dict(
            text='📊 Nivel de Confianza',
            x=0.5,
            font=dict(size=16, color='#2E7D32'),
            pad=dict(t=20)
        ),
        height=280,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Arial')
    )
    
    return fig


# Sidebar mejorado
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #4CAF50, #45a049); border-radius: 10px; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0;">⚙️ Panel de Control</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Cargar modelo
    with st.spinner("🤖 Cargando modelo inteligente..."):
        model, class_names = load_model_and_classes()
    
    if model is None or class_names is None:
        st.error("😱 No se pudo cargar el modelo. Verifica los archivos 'modelo_plantas.keras' y 'class_names.pkl'")
        st.stop()
    
    st.success("✅ ¡Modelo cargado y listo!")

    st.markdown("### 🎯 Configuración de Predicciones")
    
    show_top_n = st.slider(
        "🔝 Top predicciones a mostrar",
        min_value=3, max_value=10, value=5,
        help="Número de mejores predicciones a visualizar"
    )
    
    confidence_threshold = st.slider(
        "🎯 Umbral de confianza mínima",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="Nivel mínimo de confianza para considerar una predicción como válida"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Especificaciones del Modelo")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🌱 Clases", len(class_names))
        st.metric("🖼️ Entrada", "128x128")
    with col2:
        st.metric("🤖 Arquitectura", "VGG16")
        st.metric("📈 Precisión", "95%+")
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem; background: linear-gradient(135deg, #E8F5E8, #C8E6C9); border-radius: 8px;">
        <p style="margin: 0; font-size: 0.9rem; color: #2E7D32;">
            🌱 <b>PlantAI v2.0</b><br>
            Powered by TensorFlow & VGG16
        </p>
    </div>
    """, unsafe_allow_html=True)

# Área principal
tab1, tab2, tab3 = st.tabs(["🔍 Clasificación", "📈 Análisis", "ℹ️ Información"])

with tab1:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #2E7D32;">🔍 Clasificación Inteligente de Plantas</h2>
        <p style="color: #666; font-size: 1.1rem;">Sube una imagen y descubre qué planta es con IA avanzada</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Zona de carga de archivos mejorada
    st.markdown("""
    <div style="padding: 1rem; border: 2px dashed #4CAF50; border-radius: 10px; text-align: center; margin: 1rem 0; background: linear-gradient(135deg, #E8F5E8, #F1F8E9);">
        <h4 style="color: #2E7D32; margin: 0.5rem 0;">📤 Cargar Imagen</h4>
        <p style="color: #666; margin: 0.5rem 0;">Arrastra y suelta tu imagen aquí, o haz clic para buscar</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Selecciona una imagen de planta",
        type=['png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'],
        help="✨ Formatos soportados: PNG, JPG, JPEG, WEBP, BMP, TIFF | La imagen se redimensiona automáticamente a 128x128",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Columnas para imagen y resultados
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown("#### 🖼️ Imagen Original")
            st.image(image, caption=f"📄 {uploaded_file.name}", use_container_width=True)
            
            # Información de la imagen
            st.info(f"""
            **Detalles de la imagen:**
            - Tamaño: {image.size[0]}x{image.size[1]} px
            - Formato: {image.format}
            - Modo: {image.mode}
            """)
        
        with col2:
            # Realizar predicción
            with st.spinner("🤖 Analizando imagen con IA..."):
                predicted_class, confidence, all_predictions = predict_plant(model, image)
                predicted_name = class_names[predicted_class]
            
            # Resultado principal mejorado
            st.markdown("#### 🎯 Resultado de la Clasificación")
            
            if confidence >= confidence_threshold:
                st.markdown(f"""
                <div class="prediction-result">
                    <h3 style="color: #2E7D32; text-align: center; margin: 0;">🌱 {predicted_name}</h3>
                    <p style="text-align: center; margin: 0.5rem 0; font-size: 1.1rem;">Clasificación exitosa ✅</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #FFF3E0, #FFCC80); padding: 1.5rem; border-radius: 15px; border: 2px solid #FF9800; margin: 1rem 0;">
                    <h3 style="color: #E65100; text-align: center; margin: 0;">🤔 {predicted_name}</h3>
                    <p style="text-align: center; margin: 0.5rem 0; font-size: 1.1rem;">Confianza baja ⚠️</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            # Gráfico de confianza elegante
            confidence_fig = create_confidence_chart(confidence)
            st.plotly_chart(confidence_fig, use_container_width=True)
        
        # Gráfico de predicciones mejorado
        st.markdown("---")
        st.markdown("### 📊 Análisis Detallado de Predicciones")
        
        col_chart, col_table = st.columns([2, 1])
        
        with col_chart:
            chart_fig = create_predictions_chart(all_predictions, class_names, show_top_n)
            st.plotly_chart(chart_fig, use_container_width=True)
        
        with col_table:
            st.markdown("#### 📄 Top Predicciones")
            top_indices = np.argsort(all_predictions)[::-1][:show_top_n]
            top_data = []
            for i, idx in enumerate(top_indices):
                top_data.append({
                    'Ranking': f"🥇" if i == 0 else f"🥈" if i == 1 else f"🥉" if i == 2 else f"📍" if i == 3 else f"🪄" if i == 4 else f"{i+1}⃣",
                    'Planta': class_names[idx],
                    'Confianza': f"{all_predictions[idx]:.1%}"
                })
            
            df_top = pd.DataFrame(top_data)
            st.dataframe(df_top, use_container_width=True, hide_index=True)
        
        # Tabla completa desplegable
        with st.expander("📊 Ver todas las predicciones (tabla completa)"):
            results_df = pd.DataFrame({
                'Planta': class_names,
                'Probabilidad': all_predictions,
                'Confianza': [f"{p:.1%}" for p in all_predictions],
                'Nivel': ['Alto' if p >= 0.7 else 'Medio' if p >= 0.3 else 'Bajo' for p in all_predictions]
            })
            results_df = results_df.sort_values('Probabilidad', ascending=False).reset_index(drop=True)
            results_df.index += 1
            st.dataframe(results_df, use_container_width=True)
    
    else:
        # Estado inicial mejorado
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #E8F5E8, #F1F8E9); border-radius: 20px; border: 2px dashed #4CAF50; margin: 2rem 0;">
            <h3 style="color: #2E7D32; margin-bottom: 1rem;">🌱 ¡Listo para clasificar!</h3>
            <p style="color: #555; font-size: 1.1rem; margin-bottom: 1.5rem;">Sube una imagen de planta para comenzar el análisis</p>
            <div style="font-size: 4rem; margin: 1rem 0;">🖼️</div>
            <p style="color: #666; font-size: 0.9rem;">Arrastra y suelta tu imagen aquí ⬆️</p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #2E7D32;">📊 Dashboard del Modelo</h2>
        <p style="color: #666; font-size: 1.1rem;">Estadísticas y capacidades del sistema de IA</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #2E7D32; margin: 0;">🌱 {}</h3>
            <p style="margin: 0.5rem 0 0 0; color: #666;">Tipos de Plantas</p>
        </div>
        """.format(len(class_names)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #2E7D32; margin: 0;">🤖 VGG16</h3>
            <p style="margin: 0.5rem 0 0 0; color: #666;">Arquitectura Base</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #2E7D32; margin: 0;">🖼️ 128x128</h3>
            <p style="margin: 0.5rem 0 0 0; color: #666;">Resolución de Entrada</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #2E7D32; margin: 0;">📈 95%+</h3>
            <p style="margin: 0.5rem 0 0 0; color: #666;">Precisión</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Gráfico de distribución de clases
    st.markdown("### 🌿 Catálogo de Plantas Reconocidas")
    
    # Generar colores únicos para todas las clases
    import colorsys
    
    def generate_colors(n):
        """Generar n colores únicos y distintivos"""
        colors = []
        for i in range(n):
            hue = i / n
            saturation = 0.7 + (i % 3) * 0.1  # Varía entre 0.7 y 0.9
            lightness = 0.5 + (i % 2) * 0.2   # Varía entre 0.5 y 0.7
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            color = f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})'
            colors.append(color)
        return colors
    
    plant_colors = generate_colors(len(class_names))
    
    # Crear un gráfico de barras con todas las clases
    fig_classes = go.Figure(data=[
        go.Bar(
            x=list(range(len(class_names))),
            y=[1] * len(class_names),
            text=class_names,
            textposition='inside',
            textangle=45,
            marker=dict(
                color=plant_colors,
                line=dict(color='rgba(50,50,50,0.8)', width=1)
            ),
            hovertemplate='<b>%{text}</b><br>Clase: %{x}<extra></extra>'
        )
    ])
    
    fig_classes.update_layout(
        title=dict(
            text=f'🌱 {len(class_names)} Especies de Plantas Soportadas',
            x=0.5,
            font=dict(size=18, color='#2E7D32')
        ),
        xaxis_title='',
        yaxis_title='',
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False, showgrid=False)
    )
    
    st.plotly_chart(fig_classes, use_container_width=True)
    
    # Lista organizada de plantas
    st.markdown("#### 📋 Lista Completa de Especies")
    
    cols = st.columns(6)
    for i, plant_name in enumerate(sorted(class_names)):
        with cols[i % 6]:
            st.markdown(f"🌱 **{plant_name}**")

with tab3:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #2E7D32;">ℹ️ Acerca del Proyecto</h2>
        <p style="color: #666; font-size: 1.1rem;">Tecnología, metodología y equipo de desarrollo</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sección objetivo
    st.markdown("""
    <div style="background: linear-gradient(135deg, #E8F5E8, #C8E6C9); padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
        <h3 style="color: #1B5E20; margin-top: 0;">🎯 Objetivo del Proyecto</h3>
        <p style="font-size: 1.1rem; color: #2E7D32; line-height: 1.6;">
            PlantAI Classifier es un sistema inteligente de clasificación de plantas que utiliza 
            <strong>Deep Learning</strong> y <strong>Transfer Learning</strong> con la arquitectura VGG16. 
            El sistema es capaz de identificar <strong>{} tipos diferentes de plantas</strong> con una 
            precisión superior al 95%.
        </p>
    </div>
    """.format(len(class_names)), unsafe_allow_html=True)
    
    # Metodología con iconos
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        ### 🔧 Metodología Técnica
        
        📋 **Dataset**
        - {len(class_names)} clases de plantas
        - Miles de imágenes de alta calidad
        - Balanceado y validado
        
        🖼️ **Preprocesamiento**
        - Redimensionado a 128x128 px
        - Normalización RGB [0,1]
        - Augmentación de datos
        
        🤖 **Arquitectura**
        - VGG16 preentrenado (ImageNet)
        - Capas densas personalizadas
        - Transfer Learning optimizado
        """)
    
    with col2:
        st.markdown("""
        ### ⚙️ Configuración del Modelo
        
        🎯 **Fine-tuning**
        - Descongelado gradual de capas
        - Learning rate adaptativo
        - Optimización Adam
        
        🛡️ **Regularización**
        - Dropout (0.7) anti-overfitting
        - Early Stopping inteligente
        - Validación cruzada
        
        📊 **Métricas**
        - Precisión: 95%+
        - F1-Score balanceado
        - Matriz de confusión
        """)
    
    # Equipo de desarrollo
    st.markdown("---")
    st.markdown("### 👥 Equipo de Desarrollo")
    
    # Crear 3 columnas para el equipo
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #E8F5E8, #C8E6C9); border-radius: 15px; margin: 0.5rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">👨‍💻</div>
            <h3 style="color: #1B5E20; margin: 0.5rem 0;">Bruno Arezo</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #E3F2FD, #BBDEFB); border-radius: 15px; margin: 0.5rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">👨‍💻</div>
            <h3 style="color: #0D47A1; margin: 0.5rem 0;">Ivan Gonzalez</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #FFF3E0, #FFE0B2); border-radius: 15px; margin: 0.5rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">👩‍💻</div>
            <h3 style="color: #E65100; margin: 0.5rem 0;">Camila Pazos</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Tecnologías utilizadas
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; padding: 2rem; background: linear-gradient(135deg, #F3E5F5, #E1BEE7); border-radius: 15px;">
        <h3 style="color: #4A148C; margin-bottom: 1.5rem;">🚀 Tecnologías Utilizadas</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 1rem; max-width: 600px; margin: 0 auto;">
            <div style="background: white; padding: 0.8rem; border-radius: 15px; color: #D32F2F; font-weight: bold; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                🔥 TensorFlow
            </div>
            <div style="background: white; padding: 0.8rem; border-radius: 15px; color: #1976D2; font-weight: bold; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                📊 Streamlit
            </div>
            <div style="background: white; padding: 0.8rem; border-radius: 15px; color: #7B1FA2; font-weight: bold; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                📈 Plotly
            </div>
            <div style="background: white; padding: 0.8rem; border-radius: 15px; color: #388E3C; font-weight: bold; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                🐍 Python
            </div>
            <div style="background: white; padding: 0.8rem; border-radius: 15px; color: #F57C00; font-weight: bold; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                🧠 VGG16
            </div>
            <div style="background: white; padding: 0.8rem; border-radius: 15px; color: #5D4037; font-weight: bold; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                🖼️ PIL
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer mejorado
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f5f5f5, #e8e8e8); border-radius: 15px; margin-top: 3rem;">
    <h4 style="color: #2E7D32; margin-bottom: 1rem;">🌱 PlantAI Classifier v2.0</h4>
    <p style="color: #666; margin-bottom: 1rem;">Proyecto final Deep Learning</p>
    <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
        <span style="color: #4CAF50;">🚀 Powered by TensorFlow</span>
        <span style="color: #4CAF50;">🔍 Streamlit Dashboard</span>
        <span style="color: #4CAF50;">📊 Interactive Plotly</span>
    </div>
    <p style="color: #999; font-size: 0.8rem; margin-top: 1rem;">© 2024 - Sistema de Clasificación Inteligente de Plantas</p>
</div>
""", unsafe_allow_html=True)