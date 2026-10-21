import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import pandas as pd
import torch.nn as nn
import webbrowser  # Importar la biblioteca webbrowser

# Definir el modelo de CNN
class CNN(nn.Module):
    def __init__(self, num_classes_main, num_classes_secondary):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes_main)
        self.fc3_playa = nn.Linear(512, num_classes_secondary[0])
        self.fc3_pueblo = nn.Linear(512, num_classes_secondary[1])

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = torch.relu(self.fc1(x))
        main_out = self.fc2(x)
        playa_out = self.fc3_playa(x)
        pueblo_out = self.fc3_pueblo(x)
        return main_out, playa_out, pueblo_out

# Función para obtener detalles del lugar
def get_location_details(category, subcategory):
    location = labels_data[(labels_data['category'] == category) & (labels_data['subcategory'] == subcategory)]
    if not location.empty:
        name = location['name'].values[0]
        state = location['state'].values[0]
        return name, state
    return None, None

# Función para predecir la imagen
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Preprocesar la imagen

    with torch.no_grad():
        main_output, playa_output, pueblo_output = model(image)
        _, main_pred = torch.max(main_output, 1)

        if main_pred.item() == 0:
            _, secondary_pred = torch.max(playa_output, 1)
        else:
            _, secondary_pred = torch.max(pueblo_output, 1)

    return main_pred.item(), secondary_pred.item()

# Inicializar el modelo y cargar los pesos guardados
num_playa_classes = 356  # Número de subcategorías para playas
num_pueblo_classes = 131  # Número de subcategorías para pueblos

model = CNN(num_classes_main=2, num_classes_secondary=(num_playa_classes, num_pueblo_classes))
model.load_state_dict(torch.load('modeloCNN.pth'))
model.eval()

# Transformaciones para los datos
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Cargar los datos de etiquetas
labels_data = pd.read_csv('labels.csv')

# Función para abrir el cuadro de diálogo de selección de archivos
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        load_image(file_path)

# Función para cargar y mostrar la imagen seleccionada
def load_image(file_path):
    image = Image.open(file_path)
    image.thumbnail((300, 300))  # Redimensionar para que se ajuste a la ventana
    img = ImageTk.PhotoImage(image)

    image_label.config(image=img)
    image_label.image = img
    image_label.file_path = file_path

    # Realizar la predicción después de cargar la imagen
    category, subcategory = predict_image(file_path)
    name, state = get_location_details(category, subcategory)
    if category == 0:
        result_text.set(f"ES UNA PLAYA\nLUGAR: {name} \n ESTADO: {state}")
    else:
        result_text.set(f"ES UN PUEBLO MÁGICO\nLUGAR: {name} \n ESTADO: {state}")

# Función para mostrar la ventana "Acerca de"
def show_about():
    about_window = tk.Toplevel(root)
    about_window.title("Acerca de")
    about_window.geometry("400x300")
    about_window.resizable(False, False)  # No permitir redimensionar la ventana "Acerca de"

    info_text = """
    INFORMACIÓN RELEVANTE DEL PROYECTO:

    Nombre del proyecto: DETECCIÓN DE LUGARES TURÍSTICOS 
    A TRAVÉS DE FOTOS USANDO APRENDIZAJE PROFUNDO

    Asesor responsable: LUIS JOSE MUÑIZ RASCADO
    
    Estudiantes que participaron: 
    - MIGUEL EMILIANO ESCUDERO SAAVEDRA
    - CIELO PEREZ MUÑOZ
    """
    tk.Label(about_window, text=info_text, justify="left").pack(pady=10)

    # Simulación de un enlace
    link_label = tk.Label(about_window, text="Repositorio del proyecto", fg="blue", cursor="hand2")
    link_label.pack(pady=5)

    def open_link(event):
        webbrowser.open("https://github.com/EmiEscudero/ia-lugares-turisticos")  # Abrir el enlace en el navegador

    link_label.bind("<Button-1>", open_link)

    close_button = tk.Button(about_window, text="Cerrar", command=about_window.destroy)
    close_button.pack(pady=10)

    about_window.grab_set()  # Hacer que la ventana "Acerca de" sea modal
    center_window(about_window, 400, 300)  # Centrar la ventana "Acerca de"

# Función para centrar la ventana en la pantalla
def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    window.geometry(f"{width}x{height}+{x}+{y}")

# Crear la ventana principal
root = tk.Tk()
root.title("Detección de Lugares Turísticos")
root.geometry("800x600")
root.resizable(False, False)  # No permitir redimensionar la ventana principal

# Centrar la ventana principal
center_window(root, 800, 600)

# Espacio para mostrar la imagen
image_label = tk.Label(root)
image_label.pack(pady=20)

# Botón para cargar la imagen
upload_button = tk.Button(root, text="Subir y Evaluar Imagen", command=open_file)
upload_button.pack(pady=10)

# Espacio para mostrar el resultado de la predicción
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Arial", 14))
result_label.pack(pady=10)

# Botón para mostrar la ventana "Acerca de"
about_button = tk.Button(root, text="Acerca de", command=show_about)
about_button.pack(pady=10)

root.mainloop()
