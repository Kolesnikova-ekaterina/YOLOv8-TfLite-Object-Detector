команды:


запуск 
docker-compose build --no-cache  

выкл 
docker-compose down


TRAIN

5 эпох (быстрый запуск)
docker-compose run --rm yolo11-python train --epochs 5 --batch 4

50 эпох (полное обучение)
docker-compose run --rm yolo11-python train --epochs 50 --batch 8 --imgsz 640

проверить сохраненные веса
dir runs\train\exp{n}\weights\


VAL/TEST

тест с указанием модели
docker-compose run --rm yolo11-python test --weights runs/train/exp{n}/weights/best.pt --batch 4

val с кастомным датасетом
docker-compose run --rm yolo11-python val --weights runs/train/exp{n}/weights/best.pt --custom-data /datasets --batch 4

в докер-компоуз монтируете новый дfnfсет 

 Результаты
dir runs\test\ dir results\test\


>docker-compose run --rm yolo11-python export

