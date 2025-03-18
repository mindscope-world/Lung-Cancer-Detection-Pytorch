from django.shortcuts import render, redirect
from django.core.files.storage import default_storage
from django.conf import settings
from .utils import load_model, preprocess_image, predict_image
import os

# Loading the model
model_path = os.path.join(settings.BASE_DIR, 'predictor/Model/lung_cancer_detection_model 2.pth')
class_names = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Normal', 'Squamous Cell Carcinoma']
model = load_model(model_path, num_classes=len(class_names))

def predict(request):
    if request.method == 'POST' and request.FILES['image']:
        # Saving the uploaded image
        image_file = request.FILES['image']
        file_path = default_storage.save(image_file.name, image_file)
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)


        image_tensor = preprocess_image(full_path)
        predicted_class, probabilities = predict_image(model, image_tensor, class_names)

        default_storage.delete(file_path)

        # Passing results to the template
        context = {
            'predicted_class': predicted_class,
            'probabilities': zip(class_names, probabilities),
            'image_url': default_storage.url(file_path), 
        }
        return render(request, '/home/arichy/Documents/proj/Lung-Cancer-Detection-Pytorch/lung_cancer_app/predictor/templates/predictor/result.html', context)

    return render(request, '/home/arichy/Documents/proj/Lung-Cancer-Detection-Pytorch/lung_cancer_app/predictor/templates/predictor/upload.html')
