#!/usr/bin/env python3
import os
import sys
import shutil
import argparse
from pathlib import Path
from ultralytics import YOLO
import logging
import yaml
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv11: train/val/test/export tflite')
    parser.add_argument('command', choices=['train', 'val', 'test', 'export'], help='Команда: train/val/test/export')
    parser.add_argument('--data-dir', default='/data', help='Путь к основному датасету')
    parser.add_argument('--output-dir', default='/output', help='Путь для результатов')
    parser.add_argument('--model', default='yolo11n.pt', help='Модель')
    parser.add_argument('--epochs', type=int, default=50, help='Эпохи (для train)')
    parser.add_argument('--imgsz', type=int, default=640, help='Размер изображения')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--weights', default='runs/train/exp/weights/best.pt', help='Путь к весам модели')
    parser.add_argument('--custom-data', help='Кастомный путь для val/test')
    return parser.parse_args()

def get_device():
    """Автоопределение GPU/CPU"""
    if torch.cuda.is_available():
        device = 0
        logger.info(f"✅ GPU доступен: device={device}")
    else:
        device = 'cpu'
        logger.info("✅ Используем CPU")
    return device

def create_dataset_yaml(data_dir, custom_data_dir=None):
    yaml_path = Path('dataset.yaml')
    yaml_content = {
        'nc': 4,
        'names': ['Soap', 'liquid_soap', 'toothbrush', 'toothpaste']
    }
    
    if os.path.exists(f"{data_dir}/train"):
        yaml_content['train'] = f'{data_dir}/train/images'
    
    if custom_data_dir:
        yaml_content['val'] = f'{custom_data_dir}/images'
        yaml_content['test'] = f'{custom_data_dir}/images'
        logger.info(f"Custom dataset: {custom_data_dir}")
    else:
        if os.path.exists(f"{data_dir}/valid"):
            yaml_content['val'] = f'{data_dir}/valid/images'
        if os.path.exists(f"{data_dir}/test"):
            yaml_content['test'] = f'{data_dir}/test/images'
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)
    logger.info(f"dataset.yaml created: {yaml_path}")
    return yaml_path

def train_model(args):
    logger.info("🚀 TRAINING")
    dataset_yaml = create_dataset_yaml(args.data_dir)
    
    if not os.path.exists(f"{args.data_dir}/train"):
        logger.error(f"train/ not found: {args.data_dir}/train")
        sys.exit(1)
    
    device = get_device()
    model = YOLO(args.model)
    
    results = model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch if device == 'cpu' else min(16, args.batch),
        device=device, 
        project='runs/train',
        name='exp',
        save_period=10,
        plots=True,
        workers=2 if device == 'cpu' else 8
    )
    
    output_train = Path(args.output_dir) / 'train'
    shutil.copytree(Path('runs/train/exp'), output_train, dirs_exist_ok=True)
    logger.info(f"Train results: {output_train}")
    return Path('runs/train/exp/weights/best.pt')

def validate_model(args):
    logger.info("🔍 VALIDATION")
    dataset_yaml = create_dataset_yaml(args.data_dir, args.custom_data)
    
    custom_path = args.custom_data or f"{args.data_dir}/valid"
    if not os.path.exists(f"{custom_path}/images"):
        logger.warning(f"⚠️ images/ не найден: {custom_path}/images. Используем /data/valid")
    
    
    model_path = args.weights
    if not os.path.exists(model_path):
        logger.error(f"❌ Модель не найдена: {model_path}")
        sys.exit(1)
    
    device = get_device()
    model = YOLO(model_path)
    results = model.val(
        data=str(dataset_yaml),
        project='runs/val',
        name='valid_results',
        imgsz=args.imgsz,
        batch=args.batch if device == 'cpu' else min(16, args.batch),
        device=device,
        workers=2 if device == 'cpu' else 8
    )
    
    output_val = Path(args.output_dir) / 'validation'
    shutil.copytree(Path('runs/val/valid_results'), output_val, dirs_exist_ok=True)
    logger.info(f"✅ Val results: {output_val}")
    logger.info(f"📊 mAP50-95: {results.box.map:.4f}")
    logger.info(f"📊 mAP50: {results.box.map50:.4f}")
    logger.info(f"📊 mAP75: {results.box.map75:.4f}")

def test_model(args):
    logger.info("🧪 TESTING")
    dataset_yaml = create_dataset_yaml(args.data_dir, args.custom_data)
    
    custom_path = args.custom_data or f"{args.data_dir}/test"
    if not os.path.exists(f"{custom_path}/images"):
        logger.error(f"❌ images/ не найден: {custom_path}/images")
        sys.exit(1)
    
    model_path = args.weights
    if model_path and not os.path.exists(model_path):
        logger.error(f"❌ Модель не найдена: {model_path}")
        sys.exit(1)
    
    if not model_path:
        logger.error("❌ Укажите --weights путь к модели!")
        sys.exit(1)
    
    device = get_device()
    model = YOLO(model_path)
    
    results = model.val(
        data=str(dataset_yaml),
        project='runs/test',
        name='test_results',
        imgsz=args.imgsz,
        batch=args.batch if device == 'cpu' else min(16, args.batch),
        device=device,
        workers=2 if device == 'cpu' else 8
    )
    
    output_test = Path(args.output_dir) / 'test'
    shutil.copytree(Path('runs/test/test_results'), output_test, dirs_exist_ok=True)
    logger.info(f"✅ Test results: {output_test}")
    logger.info(f"📊 mAP50-95: {results.box.map:.4f}")
    logger.info(f"📊 mAP50: {results.box.map50:.4f}")
    logger.info(f"📊 mAP75: {results.box.map75:.4f}")
def export_model(args):
    """🔥 НОВАЯ ФУНКЦИЯ: Экспорт в TFLite"""
    logger.info("📦 EXPORTING TO TFLITE")
    
    model_path = args.weights
    if not os.path.exists(model_path):
        logger.error(f"❌ Модель не найдена: {model_path}")
        sys.exit(1)
    
    # Загрузка модели
    model = YOLO(model_path)
    
    logger.info("🔄 Экспорт в TFLite float32...")
    tflite_path = model.export(format='tflite', imgsz=args.imgsz)
    logger.info(f"✅ TFLite float32: {tflite_path}")
    
    logger.info("🔄 Экспорт в TFLite INT8 (квантование)...")
    tflite_int8_path = model.export(format='tflite', int8=True, imgsz=args.imgsz)
    logger.info(f"✅ TFLite INT8: {tflite_int8_path}")
    
    # Копируем в output
    output_export = Path(args.output_dir) / 'tflite'
    output_export.mkdir(exist_ok=True)
    
    shutil.copy(tflite_path, output_export / Path(tflite_path).name)
    shutil.copy(tflite_int8_path, output_export / Path(tflite_int8_path).name)
    
    logger.info(f"✅ TFLite модели сохранены: {output_export}")
    logger.info("📱 Готово для Android!")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Device check: CUDA={torch.cuda.is_available()}")
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'val':
        validate_model(args)
    elif args.command == 'test':
        test_model(args)
    elif args.command == 'export' :
        export_model(args)

if __name__ == "__main__":
    main()