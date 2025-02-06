import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from datetime import datetime
from ultralytics import YOLO

def train_model(model_type: str, data_dir: str, **kwargs):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=kwargs.get('batch_size', 32), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=kwargs.get('batch_size', 32), shuffle=False)

    # Возможность выбора одной из трех моделей 
    num_classes = len(train_dataset.classes)
    if model_type == 'resnet':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == 'vgg':
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_type == 'efficientnet':
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Unsupported model type. Choose from 'resnet', 'vgg', or 'efficientnet'.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=kwargs.get('lr', 0.001))

    # Выполняет обучение алгоритма машинного обучения на обучающем наборе данных
    epochs = kwargs.get('epochs', 10)
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

    training_time = time.time() - start_time

    # Проводит оценку точности на валидационной выборке
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    roc_auc = roc_auc_score(all_labels, all_preds, multi_class='ovr')

    metrics = {
        'accuracy': accuracy,
        'roc_auc_score': roc_auc,
        'f1_score_macro': f1
    }

    # 6. Подготовка результатов
    results = {
        'model': model,
        'metrics': metrics,
        'training_time_per_epoch': training_time / epochs,
        'total_training_time_minutes': training_time / 60
    }

    # 7. Сохранение результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"model_results_{timestamp}.pt"
    torch.save(results, save_path)

    return results

def get_folder_names_and_save(folder_path, output_file="classes.txt"):
    """
    Функция принимает путь к папке, получает имена папок внутри неё,
    и записывает их в файл в формате 'names: [...]'.

    :param folder_path: Путь к главной папке
    :param output_file: Имя файла для записи результата (по умолчанию 'classes.txt')
    """
    # Проверяем, существует ли путь и является ли он папкой
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Путь '{folder_path}' не существует.")
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Путь '{folder_path}' не является папкой.")

    # Получаем список всех элементов в папке
    all_items = os.listdir(folder_path)

    # Фильтруем только папки
    folder_names = [item for item in all_items if os.path.isdir(os.path.join(folder_path, item))]

    # Формируем результат в требуемом формате
    result = {"names": folder_names}

    # Записываем результат в файл
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(f"names: {result['names']}")

    print(f"Результат успешно записан в файл '{output_file}'.")

def train_yolo_model(**kwargs):
    # Извлечение имени модели из kwargs
    model_name = kwargs.pop('model_name')  # Удаляем model_name из kwargs и сохраняем в переменной
    
    # Возможность выбора одной из трех моделей
    if model_name in ["yolov8n.pt", "yolov8m.pt", "yolov8s.pt"]:
        print("Переменная равна одному из значений.")
    
        print("Переменная не равна ни одному из значений.")
        # Загрузка модели
        model = YOLO(model_name)
        
        # Обучение модели
        start_time = time.time()
        results = model.train(
            data='custom_data.yaml',
            imgsz=kwargs.get('imgsz', 640),  # Параметр по умолчанию 640, если не указан
            epochs=kwargs.get('epochs', 50),  # Параметр по умолчанию 50, если не указан
            batch=kwargs.get('batch', 16),    # Параметр по умолчанию 16, если не указан
            name='yolov8n_custom'
        )
        # Засекаем конечное время
        end_time = time.time()

        # Вычисляем разницу во времени (время выполнения функции)
        execution_time = end_time - start_time
        # Сохранение модели
        saved_model_path = model.export(format='onnx')  # Пример сохранения в формате ONNX
        print(f"Модель сохранена по пути: {saved_model_path}")
        
        # Запись имени модели в файл log.txt
        with open('log.txt', 'a') as log_file:  # Открываем файл в режиме добавления ('a')
            log_file.write(f"Model Name: {model_name}, Saved Path: {saved_model_path}\n")
        print(f"Model Name: {model_name}, Saved Path: {saved_model_path}\n")
        
        # запись время выполнения
        with open('log.txt', 'a') as file:
            file.write(f'Время выполнения функции: {execution_time:.4f} секунд\n')
        file.print(f'Время выполнения функции: {execution_time:.4f} секунд\n')
    
        return results
    else:
        return 0