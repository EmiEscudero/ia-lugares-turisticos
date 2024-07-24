import argparse
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import torch.nn as nn

# Transformaciones para los datos
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Cargar los datos de etiquetas
labels_data = pd.read_csv('labels.csv')

# Función para obtener detalles del lugar
def get_location_details(category, subcategory):
    location = labels_data[(labels_data['category'] == category) & (labels_data['subcategory'] == subcategory)]
    if not location.empty:
        name = location['name'].values[0]
        state = location['state'].values[0]
        return name, state
    return None, None

# Red neuronal convolucional
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

# Inicializar el modelo y cargar los pesos guardados
num_playa_classes = 356  # Número de subcategorías para playas
num_pueblo_classes = 131  # Número de subcategorías para pueblos

model = CNN(num_classes_main=2, num_classes_secondary=(num_playa_classes, num_pueblo_classes))
model.load_state_dict(torch.load('modeloCNN.pth'))
model.eval()

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

def main():
    parser = argparse.ArgumentParser(description='Predict the category and subcategory of a tourist location image.')
    parser.add_argument('image_path', type=str, help='Path to the image file to be predicted.')

    args = parser.parse_args()

    # Realizar predicción
    category, subcategory = predict_image(args.image_path)
    name, state = get_location_details(category, subcategory)
    
    if category == 0:
        print("Categoría: PLAYA")
    else:
        print("Categoría: PUEBLO MÁGICO")
    
    print(f"Lugar: {name}, Estado: {state}")

if __name__ == '__main__':
    main()

