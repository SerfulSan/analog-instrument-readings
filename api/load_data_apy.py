import os
from PIL import Image
import shutil
import cv2
import os
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def extract_archive(archive_path):
    # Определяем путь к директории, куда будет распакован архив
    target_dir = os.path.dirname(archive_path)  # Распаковываем в ту же директорию, где находится архив
    
    # Временная директория для распаковки
    temp_dir = os.path.join(target_dir, 'temp_extract')
    
    try:
        # Распаковываем архив во временную директорию
        shutil.unpack_archive(archive_path, temp_dir)
        
        # Получаем список содержимого временной директории
        contents = os.listdir(temp_dir)
        
        # Проверяем, есть ли одна главная папка (например, Gauge_big)
        if len(contents) == 1 and os.path.isdir(os.path.join(temp_dir, contents[0])):
            main_folder = os.path.join(temp_dir, contents[0])
            
            # Переименовываем главную папку в 'images'
            images_dir = os.path.join(target_dir, 'images')
            if os.path.exists(images_dir):
                shutil.rmtree(images_dir)  # Удаляем старую папку images, если она существует
            
            shutil.move(main_folder, images_dir)
            print(f"Архив успешно распакован директорию '{images_dir}'.")
        else:
            print("В архиве не найдено ровно одной главной папки.")
        
        # Удаляем временную директорию
        shutil.rmtree(temp_dir)
    
    except Exception as e:
        print(f"Ошибка при распаковке или обработке архива: {e}")


def process_images(input_dir, output_dir="preprocessed_images", max_count=0):
    """
    Функция находит файлы в указанной папке, изменяет их размер до 224x224,
    преобразует в оттенки серого и сохраняет результат в указанную директорию
    с сохранением структуры поддиректорий.

    :param directory: Путь к папке, в которой нужно найти файлы.
    :param input_dir: Исходная директория, где находятся изображения.
    :param output_dir: Директория для сохранения обработанных изображений.
    :param max_count: Максимальное количество файлов для обработки. Если 0, обрабатываются все файлы.
    :return: Список путей к сохраненным изображениям.
    """
    if not os.path.isdir(input_dir):
        raise ValueError(f"Указанный путь '{input_dir}' не является директорией.")

    file_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            # Если max_count != 0 и достигнуто ограничение, прекращаем поиск
            if max_count > 0 and len(file_paths) >= max_count:
                break
            file_paths.append(os.path.join(root, file))
        if max_count > 0 and len(file_paths) >= max_count:
            break

    # Обрезаем список файлов до max_count, если max_count > 0
    if max_count > 0:
        file_paths = file_paths[:max_count]

    processed_files = []
    
    for file_path in file_paths:
        # Вычисляем относительный путь файла относительно input_dir
        relative_path = os.path.relpath(file_path, input_dir)

        try:
            # Полный путь к исходному изображению
            full_input_path = os.path.join(input_dir, relative_path)

            # Проверяем, существует ли файл
            if not os.path.isfile(full_input_path):
                raise FileNotFoundError(f"Файл не найден: {full_input_path}")

            # Открываем изображение
            img = Image.open(full_input_path)

            # Изменяем размер изображения до 224x224
            img_resized = img.resize((224, 224))

            # Преобразуем изображение в оттенки серого
            img_gray = img_resized.convert("L")

            # Создаем путь для сохранения изображения
            output_path = os.path.join(output_dir, relative_path)

            # Создаем директории, если они не существуют
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Сохраняем изображение
            img_gray.save(output_path)

            processed_files.append(output_path)
        except Exception as e:
            print(f"Ошибка при обработке файла {file_path}: {e}")

    return processed_files

def process_images2(input_dir, output_dir="preprocessed_images", max_count=0):
    """ Функция находит файлы в указанной папке, изменяет их размер до 224x224, преобразует в оттенки серого и сохраняет результат в указанную директорию с сохранением структуры поддиректорий. :param input_dir: Исходная директория, где находятся изображения. :param output_dir: Директория для сохранения обработанных изображений. :param max_count: Максимальное количество файлов для обработки. Если 0, обрабатываются все файлы. :return: Список путей к сохраненным изображениям. """
    if not os.path.isdir(input_dir):
        raise ValueError(f"Указанный путь '{input_dir}' не является директорией.")

    file_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            # Если max_count != 0 и достигнуто ограничение, прекращаем поиск
            if max_count > 0 and len(file_paths) >= max_count:
                break
            file_paths.append(os.path.join(root, file))
        if max_count > 0 and len(file_paths) >= max_count:
            break

    # Обрезаем список файлов до max_count, если max_count > 0
    if max_count > 0:
        file_paths = file_paths[:max_count]

    processed_files = []
    
    for file_path in file_paths:
        # Вычисляем относительный путь файла относительно input_dir
        relative_path = os.path.relpath(file_path, input_dir)

        try:
            # Полный путь к исходному изображению
            full_input_path = os.path.join(input_dir, relative_path)

            # Проверяем, существует ли файл
            if not os.path.isfile(full_input_path):
                raise FileNotFoundError(f"Файл не найден: {full_input_path}")

            # Читаем изображение в оттенках серого
            img = cv2.imread(full_input_path, cv2.IMREAD_GRAYSCALE)
            
            # Изменяем размер изображения до 224x224
            img_resized = cv2.resize(img, (224, 224))

            # Создаем путь для сохранения изображения
            output_path = os.path.join(output_dir, relative_path)

            # Создаем директории, если они не существуют
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Сохраняем изображение
            cv2.imwrite(output_path, img_resized)

            processed_files.append(output_path)
        except Exception as e:
            print(f"Ошибка при обработке файла {file_path}: {e}")

    return processed_files

def process_image(image_path, show_results=False):

    img_orig = Image.open(image_path)

    # Открываем изображение
    with Image.open(image_path) as img:
        # Получаем размеры изображения
        width, height = img.size

        # Генерируем случайный угол поворота от -15 до +15 градусов
        angle = random.uniform(-15, 15)
        
        # Поворачиваем изображение
        tilted_img = img.rotate(angle, expand=False)

        # Генерируем случайный размер прямоугольника
        rect_width = random.randint(20, 50)
        rect_height = random.randint(20, 50)

        # Генерируем случайные координаты для верхнего левого угла прямоугольника
        x = random.randint(0, width - rect_width)
        y = random.randint(0, height - rect_height)

        # Создаем черный прямоугольник
        draw = ImageDraw.Draw(img)
        draw.rectangle([x, y, x + rect_width, y + rect_height], fill="black")

        # Формируем имена файлов для сохранения
        base_name, ext = os.path.splitext(image_path)
        
        if show_results:
            # Создаем график с тремя полотнами
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            
            # Выводим исходное изображение
            axs[0].imshow(img_orig)
            axs[0].set_title("Исходное изображение")

            # Выводим изображение с поворотом
            axs[1].imshow(tilted_img)
            axs[1].set_title("Поворот (градусов: {:.2f})".format(angle))

            # Выводим изображение с черным прямоугольником
            axs[2].imshow(img)
            axs[2].set_title("Черный прямоугольник")
            
            # Добавляем легенду
            plt.tight_layout()
            plt.show()
        else:
            base_corrupted_file = base_name + "_corrupted" + ext
            base_tilted_file = base_name + "_tilted" + ext
            if not os.path.exists(base_corrupted_file):
                img.save(base_corrupted_file)
            else:
                print(f"_corrupted файл уже существует. НЕ СОХРАНЯЕМ")

            if not os.path.exists(base_tilted_file):
                tilted_img.save(base_tilted_file)
            else:
                print(f"_tilted файл уже существует. НЕ СОХРАНЯЕМ")

def process_images_one_in_all(base_dir="preprocessed_images/train"):
    # Проходим по каждой папке (классу) в базовой директории
    for class_dir in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_dir)

        # Убедимся, что мы имеем дело с директорией
        if os.path.isdir(class_path):
            # Находим все файлы в папке класса
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            # Если есть хотя бы одно изображение, обрабатываем первое
            if image_files:
                # Берем первое изображение
                first_image_path = os.path.join(class_path, image_files[0])
                # Применяем функцию обработки к изображению
                process_image(first_image_path, show_results=False)
            #print(class_path)

def copy_images(input_directory, output_directory, num_images):
    # Создаем выходную директорию, если она не существует
    os.makedirs(output_directory, exist_ok=True)

    # Список для хранения найденных изображений
    image_files = []
    # Расширения файлов изображений
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

    # Проходим по всем подкаталогам и файлам в указанной директории
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            # Проверяем, является ли файл изображением по его расширению
            if file.lower().endswith(valid_extensions):
                image_files.append(os.path.join(root, file))
            # Если уже собрали достаточное количество изображений, выходим из цикла
            if len(image_files) >= num_images:
                break
        if len(image_files) >= num_images:
            break

    # Копируем найденные изображения в целевую директорию
    for image in image_files[:num_images]:  # Копируем только указанное количество
        shutil.copy(image, output_directory)

    print(f'Скопировано {len(image_files[:num_images])} изображений в {output_directory}')

def zip_folder(folder_name):
    # Проверяем, существует ли директория
    if not os.path.exists(folder_name):
        print(f"Директория '{folder_name}' не существует.")
        return

    # Формируем имя для выходного ZIP файла
    zip_file_name = 'preprocessed_images.zip'

    # Удаляем файл, если он уже существует
    if os.path.exists(zip_file_name):
        os.remove(zip_file_name)

    # Создаем архив
    shutil.make_archive(zip_file_name.replace('.zip', ''), 'zip', folder_name)

    print(f"Архив '{zip_file_name}' успешно создан.")

def print_directory_structure(path, indent=0, is_last=True, prefix="", show_files_limit=6):
    # Проверяем, существует ли указанный путь
    if not os.path.exists(path):
        print(f"Путь '{path}' не существует.")
        return

    # Получаем список всех элементов в директории
    try:
        items = os.listdir(path)
    except PermissionError:
        print(prefix + '└── Доступ запрещен к этой папке.')
        return

    files = [item for item in items if os.path.isfile(os.path.join(path, item))]
    dirs = [item for item in items if os.path.isdir(os.path.join(path, item))]

    # Выводим название директории с отступами и количеством файлов в ней
    connector = "└── " if is_last else "├── "
    print(prefix + connector + f"{os.path.basename(path)}/ ({len(files)} файлов)")

    # Обновляем префикс для вложенных элементов
    if is_last:
        new_prefix = prefix + "    "
    else:
        new_prefix = prefix + "│   "

    # Обрабатываем директории
    for i, directory in enumerate(dirs):
        is_last_dir = (i == len(dirs) - 1) and (len(files) == 0)
        print_directory_structure(os.path.join(path, directory), indent + 4, is_last_dir, new_prefix)

    # Обрабатываем файлы
    if len(files) > show_files_limit:
        # Если в папке больше 6 файлов, отображаем первые и последние 3 файла
        print(new_prefix + "└── ... (еще {0} файлов)".format(len(files) - 6))
        for file in files[:3]:
            print(new_prefix + "    ├── " + file)
        if len(files) > 6:  # Если больше 6 файлов, показываем последние 3
            for file in files[-3:]:
                print(new_prefix + "    ├── " + file)
    else:
        for i, file in enumerate(files):
            connector = "└── " if (i == len(files) - 1) else "├── "
            print(new_prefix + connector + file)