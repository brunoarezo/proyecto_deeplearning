import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import logging
import time
from PIL import Image
from typing import List, Tuple
import pandas as pd
import os
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TODO: Replace with your actual class names
CLASS_NAMES = [
    "clase_1", "clase_2", "clase_3", "clase_4", "clase_5",
    "clase_6", "clase_7", "clase_8", "clase_9", "clase_10"
]

# Alternative: Load class names from file if available
def load_class_names() -> List[str]:
    """Load class names from classes.txt if available, otherwise use default."""
    if os.path.exists('classes.txt'):
        try:
            with open('classes.txt', 'r') as f:
                return [line.strip() for line in f.readlines()]
        except Exception as e:
            logger.warning(f"Could not load classes.txt: {e}")
    return CLASS_NAMES

@st.cache_resource
def load_model(model_path: str = "model.h5") -> tf.keras.Model:
    """Load the pre-trained VGG16 model with caching."""
    start_time = time.time()
    try:
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"GPU available: {len(gpus)} device(s)")
        else:
            logger.info("No GPU available, using CPU")
        
        if not os.path.exists(model_path):
            st.error(f"Model file '{model_path}' not found. Please ensure your trained model is in the current directory.")
            st.info("**Instrucciones:**\n1. Asegúrate de que tu archivo `model.h5` esté en el directorio actual\n2. O cambia la ruta en la función `load_model()`")
            st.stop()
        
        model = tf.keras.models.load_model(model_path)
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f}s")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error(f"Error loading model: {e}")
        st.info("**Posibles soluciones:**\n- Verifica que el archivo de modelo sea válido\n- Asegúrate de tener TensorFlow instalado correctamente")
        st.stop()

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for VGG16 model.
    
    Args:
        image: PIL Image
        
    Returns:
        Preprocessed image array ready for model input
    """
    try:
        # Resize to 224x224
        image = image.resize((224, 224))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply VGG16 preprocessing
        img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

@st.cache_data
def predict_image(_model: tf.keras.Model, img_array: np.ndarray, 
                 _img_hash: str) -> np.ndarray:
    """
    Make prediction on preprocessed image with caching.
    
    Args:
        _model: Loaded Keras model
        img_array: Preprocessed image array
        img_hash: Hash of the original image for caching
        
    Returns:
        Prediction probabilities array
    """
    start_time = time.time()
    try:
        predictions = _model.predict(img_array, verbose=0)
        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.3f}s")
        return predictions[0]
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

def get_top_predictions(predictions: np.ndarray, class_names: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Get top-k predictions with class names and probabilities.
    
    Args:
        predictions: Model prediction probabilities
        class_names: List of class names
        top_k: Number of top predictions to return
        
    Returns:
        List of (class_name, probability) tuples
    """
    top_indices = np.argsort(predictions)[::-1][:top_k]
    return [(class_names[i], float(predictions[i])) for i in top_indices]

def generate_gradcam(model: tf.keras.Model, img_array: np.ndarray, class_index: int, 
                    last_conv_layer_name: str = None) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for the predicted class.
    
    Args:
        model: Trained model
        img_array: Preprocessed image
        class_index: Index of the class to generate CAM for
        last_conv_layer_name: Name of the last convolutional layer
        
    Returns:
        Heatmap array
    """
    try:
        # Find the last convolutional layer if not specified
        if last_conv_layer_name is None:
            for layer in reversed(model.layers):
                if len(layer.output_shape) == 4:
                    last_conv_layer_name = layer.name
                    break
        
        if last_conv_layer_name is None:
            raise ValueError("No convolutional layer found in model")
        
        # Create a model that maps the input image to the activations of the last conv layer
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        # Compute the gradient of the top predicted class for our input image
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_index]
        
        # Extract the gradients of the top predicted class with regard to the output feature map
        grads = tape.gradient(loss, conv_outputs)
        
        # Pool the gradients over all the axes leaving out the channel dimension
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply each channel in the feature map array by "how important this channel is"
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    except Exception as e:
        logger.error(f"Error generating Grad-CAM: {e}")
        return None

def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """
    Overlay heatmap on original image.
    
    Args:
        image: Original image (224x224x3)
        heatmap: Grad-CAM heatmap
        alpha: Transparency of heatmap overlay
        
    Returns:
        Image with heatmap overlay
    """
    try:
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        
        # Convert heatmap to RGB using colormap
        jet = plt.cm.get_cmap('jet')
        heatmap_colored = jet(heatmap_resized)[:, :, :3]  # Remove alpha channel
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Normalize original image
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Overlay heatmap on image
        overlaid = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
        
        return overlaid
    except Exception as e:
        logger.error(f"Error overlaying heatmap: {e}")
        return image

def validate_image(uploaded_file) -> bool:
    """Validate uploaded image file."""
    if uploaded_file is None:
        return False
    
    # Check file type
    if uploaded_file.type not in ['image/jpeg', 'image/jpg', 'image/png']:
        st.error("Formato no válido. Por favor sube una imagen en formato JPG o PNG.")
        return False
    
    # Check file size (max 10MB)
    if uploaded_file.size > 10 * 1024 * 1024:
        st.error("La imagen es demasiado grande. Máximo 10MB.")
        return False
    
    return True

def process_batch_images(uploaded_files: List, model: tf.keras.Model, 
                        class_names: List[str], top_k: int) -> pd.DataFrame:
    """Process multiple images and return results as DataFrame."""
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f'Procesando {uploaded_file.name}...')
            
            # Load and preprocess image
            image = Image.open(uploaded_file)
            img_array = preprocess_image(image)
            img_hash = str(hash(uploaded_file.getvalue()))
            
            # Make prediction
            predictions = predict_image(model, img_array, img_hash)
            top_preds = get_top_predictions(predictions, class_names, top_k)
            
            # Store results
            result = {
                'filename': uploaded_file.name,
                'top1_class': top_preds[0][0],
                'top1_prob': top_preds[0][1]
            }
            
            # Add top-k classes and probabilities
            top_classes = [pred[0] for pred in top_preds]
            top_probs = [pred[1] for pred in top_preds]
            result['top3_classes'] = ', '.join(top_classes)
            result['top3_probs'] = ', '.join([f'{prob:.4f}' for prob in top_probs])
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing {uploaded_file.name}: {e}")
            results.append({
                'filename': uploaded_file.name,
                'top1_class': 'Error',
                'top1_prob': 0.0,
                'top3_classes': 'Error',
                'top3_probs': 'Error'
            })
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text('¡Procesamiento completado!')
    return pd.DataFrame(results)

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Clasificador de Plantas",
        page_icon="🌱",
        layout="wide"
    )
    
    st.title("🌱 Clasificador de Plantas con VGG16")
    st.write("Sube una imagen de planta para obtener su clasificación con visualización Grad-CAM")
    
    # Load class names and model
    class_names = load_class_names()
    model = load_model()
    
    # Sidebar controls
    st.sidebar.header("⚙️ Configuración")
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Umbral de confianza", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.1, 
        step=0.05
    )
    
    # Top-K predictions
    top_k = st.sidebar.selectbox(
        "Número de predicciones (Top-K)", 
        options=[1, 2, 3, 4, 5], 
        index=2
    )
    
    # Grad-CAM alpha
    gradcam_alpha = st.sidebar.slider(
        "Transparencia Grad-CAM", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.6, 
        step=0.1
    )
    
    # Batch processing toggle
    batch_mode = st.sidebar.checkbox("Procesamiento en lote")
    
    # Clear button
    if st.sidebar.button("🗑️ Limpiar"):
        st.rerun()
    
    # File uploader
    if batch_mode:
        uploaded_files = st.file_uploader(
            "Selecciona múltiples imágenes",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Arrastra y suelta múltiples imágenes aquí"
        )
        
        if uploaded_files:
            st.write(f"📁 {len(uploaded_files)} archivos seleccionados")
            
            if st.button("🚀 Procesar todas las imágenes"):
                with st.spinner("Procesando imágenes..."):
                    results_df = process_batch_images(uploaded_files, model, class_names, top_k)
                
                st.success("¡Procesamiento completado!")
                st.dataframe(results_df)
                
                # Download CSV
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar resultados (CSV)",
                    data=csv,
                    file_name="predicciones_plantas.csv",
                    mime="text/csv"
                )
    
    else:
        uploaded_file = st.file_uploader(
            "Selecciona una imagen",
            type=['png', 'jpg', 'jpeg'],
            help="Arrastra y suelta una imagen aquí"
        )
        
        if uploaded_file is not None:
            if validate_image(uploaded_file):
                # Display original image
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📷 Imagen Original")
                    image = Image.open(uploaded_file)
                    st.image(image, use_column_width=True)
                
                with col2:
                    try:
                        # Preprocess image
                        img_array = preprocess_image(image)
                        img_hash = str(hash(uploaded_file.getvalue()))
                        
                        # Make prediction
                        with st.spinner("Analizando imagen..."):
                            predictions = predict_image(model, img_array, img_hash)
                        
                        # Get top predictions
                        top_preds = get_top_predictions(predictions, class_names, top_k)
                        
                        # Display predictions
                        st.subheader("🎯 Predicciones")
                        
                        for i, (class_name, prob) in enumerate(top_preds):
                            if prob >= confidence_threshold:
                                st.write(f"**#{i+1}: {class_name}**")
                                st.progress(prob)
                                st.write(f"Confianza: {prob:.4f} ({prob*100:.2f}%)")
                                st.write("---")
                        
                        # Generate and display Grad-CAM
                        st.subheader("🔍 Visualización Grad-CAM")
                        
                        with st.spinner("Generando Grad-CAM..."):
                            # Use top prediction for Grad-CAM
                            top_class_idx = np.argmax(predictions)
                            heatmap = generate_gradcam(model, img_array, top_class_idx)
                        
                        if heatmap is not None:
                            # Prepare image for overlay (denormalize)
                            display_img = img_array[0].copy()
                            display_img = (display_img - display_img.min()) / (display_img.max() - display_img.min())
                            display_img = (display_img * 255).astype(np.uint8)
                            
                            # Create overlay
                            overlaid_img = overlay_heatmap(display_img, heatmap, gradcam_alpha)
                            
                            st.image(
                                overlaid_img, 
                                caption=f"Grad-CAM para: {class_names[top_class_idx]}", 
                                use_column_width=True
                            )
                            
                            # Show explanation
                            st.info(
                                "🔍 Las áreas rojas/amarillas indican las regiones más importantes "
                                "para la clasificación según el modelo."
                            )
                        else:
                            st.warning("No se pudo generar la visualización Grad-CAM")
                    
                    except Exception as e:
                        st.error(f"Error procesando la imagen: {e}")
                        logger.error(f"Error in main processing: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🤖 **Modelo:** VGG16 Transfer Learning | "
        "📊 **Framework:** TensorFlow/Keras | "
        "🎨 **UI:** Streamlit"
    )

if __name__ == "__main__":
    main()