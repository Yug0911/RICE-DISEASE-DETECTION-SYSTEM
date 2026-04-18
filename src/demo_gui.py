"""
Rice Leaf Disease Detection - Interactive Demo
Upload an image and get instant prediction
"""
import os
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Config
MODEL_PATH = "models/best_5class.h5"
IMG_SIZE = (300, 300)
CLASS_NAMES = ['Bacterialblight', 'Brownspot', 'Healthy', 'Leafsmut', 'Rice Blast']

# Load model once at startup
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded. Ready for predictions.")

def predict_image(img_path):
    """Load image, preprocess, predict"""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_class_idx = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_class_idx]
    confidence = predictions[predicted_class_idx] * 100
    
    # Get all class probabilities
    all_probs = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}
    return predicted_class, confidence, all_probs

# GUI
class RiceDiseaseDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("Rice Leaf Disease Detection - Demo")
        self.root.geometry("700x600")
        self.root.configure(bg='#f0f0f0')
        
        # Header
        header = tk.Label(root, text="Rice Leaf Disease Detection", 
                         font=("Arial", 20, "bold"), bg='#2c7a7b', fg='white', pady=20)
        header.pack(fill='x')
        
        # Upload button
        upload_btn = tk.Button(root, text="📁 Upload Leaf Image", 
                               command=self.upload_image,
                               font=("Arial", 14), bg='#4299e1', fg='white',
                               padx=20, pady=10, cursor='hand2')
        upload_btn.pack(pady=20)
        
        # Image display
        self.image_label = tk.Label(root, bg='#f0f0f0')
        self.image_label.pack(pady=10)
        
        # Results frame
        result_frame = tk.Frame(root, bg='white', relief='ridge', bd=2)
        result_frame.pack(pady=20, padx=50, fill='both', expand=True)
        
        tk.Label(result_frame, text="Prediction Results", 
                font=("Arial", 16, "bold"), bg='white').pack(pady=10)
        
        self.result_label = tk.Label(result_frame, text="No image selected", 
                                     font=("Arial", 12), bg='white', fg='#666')
        self.result_label.pack(pady=5)
        
        self.confidence_label = tk.Label(result_frame, text="", 
                                         font=("Arial", 11), bg='white')
        self.confidence_label.pack(pady=5)
        
        # All probabilities
        self.prob_frame = tk.Frame(result_frame, bg='white')
        self.prob_frame.pack(pady=10, padx=20, fill='both')
        
        self.prob_labels = {}
        for i, cls in enumerate(CLASS_NAMES):
            lbl = tk.Label(self.prob_frame, text=f"{cls}: --", 
                          font=("Consolas", 10), bg='white', anchor='w')
            lbl.pack(anchor='w', pady=2)
            self.prob_labels[cls] = lbl
        
        # Footer
        footer = tk.Label(root, text="Model Accuracy: 95.23% | 5-Class Balanced Dataset", 
                         font=("Arial", 9), bg='#f0f0f0', fg='#666', pady=10)
        footer.pack(side='bottom')
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Rice Leaf Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        if not file_path:
            return
        
        try:
            # Display image
            img = Image.open(file_path)
            img.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(img)
            self.image_label.configure(image=photo)
            self.image_label.image = photo
            
            # Predict
            predicted_class, confidence, all_probs = predict_image(file_path)
            
            # Update result
            color = '#2c7a7b' if confidence > 80 else '#c53030' if confidence < 60 else '#d69e2e'
            self.result_label.configure(
                text=f"Prediction: {predicted_class}",
                font=("Arial", 16, "bold"),
                fg=color
            )
            self.confidence_label.configure(
                text=f"Confidence: {confidence:.1f}%",
                fg=color
            )
            
            # Update probabilities
            for cls, prob in all_probs.items():
                bar_len = int(prob * 40)
                bar = '█' * bar_len + '░' * (40 - bar_len)
                pct = prob * 100
                self.prob_labels[cls].configure(
                    text=f"{cls:<20}: {bar} {pct:>5.1f}%"
                )
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = RiceDiseaseDemo(root)
    root.mainloop()
