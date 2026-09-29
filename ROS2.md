# Загрузка и обработка LiDAR в ROS 2 Humble

Окружение контейнера: **Ubuntu 22.04 Jammy, ROS 2 Humble, Python 3.10**.
Пакет `obstacle_detector_ros` собирается через `ament_python` / `colcon`.
Алгоритм `obstacle_detector` включён в устанавливаемый пакет; отдельная копия
алгоритма или пути к рабочей папке разработчика не нужны. Габарит — 2,1 × 3,0 м.

```text
DB3 / папка записи / NPY / NPZ
             │
        lidar_loader
             │  /lidar_points  (sensor_msgs/msg/PointCloud2)
     obstacle_processor  ← также принимает топик внешнего лидара
             ├─ /obstacle_detector/result     (std_msgs/msg/String, JSON)
             ├─ /obstacle_detector/corridor   (sensor_msgs/msg/PointCloud2)
             └─ /obstacle_detector/obstacles  (sensor_msgs/msg/PointCloud2)
```

## Сборка Docker

Из корня этого репозитория:

```bash
docker build -t obstacle_detector:humble .
docker build --target test -t obstacle_detector:humble-test .
```

Вторая команда дополнительно запускает тесты внутри Humble: реальные ROS-сообщения,
DDS-публикации и подписки, обработку ошибок, обратное подтверждение каждого кадра,
установленный `ros2 launch`, DB3/NPZ и завершение записи. Ошибка теста прерывает сборку.
Workflow `.github/workflows/ros2.yml` выполняет обе сборки на GitHub Actions при push/PR.
После успешных тестов workflow сохраняет готовый образ в артефакт
`obstacle-detector-humble` (файл `obstacle_detector_humble.tar.gz`, хранится 7 дней).
Его можно скачать из завершённого запуска Actions и загрузить через `docker load -i`.
В registry образ этим workflow не публикуется.

Без аргументов контейнер показывает параметры launch:

```bash
docker run --rm obstacle_detector:humble
docker run --rm obstacle_detector:humble ros2 pkg executables obstacle_detector_ros
```

Если организаторы принимают Docker-образ файлом:

```bash
docker save -o obstacle_detector_humble.tar obstacle_detector:humble
# На другом компьютере:
docker load -i obstacle_detector_humble.tar
```

## Запись: обе ноды одной командой

PowerShell на Windows с работающим Docker Desktop в режиме Linux-контейнеров:

```powershell
docker run --rm `
  -v "D:\input:/data:ro" `
  -v "D:\output:/output" `
  obstacle_detector:humble `
  ros2 launch obstacle_detector_ros pipeline.launch.py `
  input:=/data/example.db3 `
  output_jsonl:=/output/ros2_results.jsonl
```

Linux:

```bash
docker run --rm -v "$PWD/data:/data:ro" -v "$PWD/output:/output" \
  obstacle_detector:humble ros2 launch obstacle_detector_ros pipeline.launch.py \
  input:=/data/recording output_jsonl:=/output/ros2_results.jsonl
```

`input` принимает один `.db3`, папку с частями **одной** записи, `.npy` или `.npz`.
NPY содержит XYZ формы `(N, 3)`; NPZ — массив `xyz` и необязательный целочисленный
`ring` длины N. Для нескольких `PointCloud2`-топиков в записи задайте
`bag_topic:=/lidar_points`. Для проверки первых кадров добавьте `max_frames:=20`.
`cloud_topic` задаёт имя публикуемого топика и входного топика обработчика.

Загрузчик ждёт появления обработчика, публикует один кадр и ждёт его результата.
Следующий кадр не вытеснит предыдущий из очереди во время компиляции Numba.
`rate_hz:=10.0` ограничивает максимальную скорость подачи; медленная обработка
замедляет воспроизведение. Это последовательная обработка, а не гарантия реального времени.
Ожидание подключения и результата ограничено `timeout_sec:=300.0`, с запасом
на первоначальную компиляцию Numba. Для прогретого процесса его можно уменьшить.

Поля и байты облака DB3 сохраняются при нативной десериализации. В `header.stamp`
ставится **время записи сообщения из DB3**, как в прежнем файловом обработчике.
Исходный `frame_id` сохраняется; пустой заменяется на `lidar`. NPY/NPZ получают
время публикации и `frame_id=lidar`. Параметр `frame_id` позволяет переименовать
систему координат, но **не преобразует координаты**. Данные задаются в метрах:
**вперёд −Y, вверх +Z**. Для иной ориентации нужен внешний трансформатор облака.

После результата последнего кадра обе ноды завершаются. Ошибка загрузчика или
процесса обработки приводит к ненулевому коду завершения launch.
Ошибка конкретного облака публикуется как `processing_ok: false`, `status: CAUTION`,
`path_clear: null`; загрузчик останавливает воспроизведение с ошибкой.

## Отдельные ноды и внешний лидар

В двух терминалах подготовленного ROS-окружения:

```bash
ros2 run obstacle_detector_ros obstacle_processor --ros-args \
  -p cloud_topic:=/lidar_points -p output_jsonl:=/tmp/detection.jsonl
ros2 run obstacle_detector_ros lidar_loader --ros-args \
  -p input:=/data/recording -p bag_topic:=/lidar_points
```

Обработка внешнего топика с сенсорным QoS:

```bash
ros2 launch obstacle_detector_ros pipeline.launch.py \
  start_loader:=false cloud_topic:=/lidar_points input_reliability:=best_effort
```

Для подключения контейнера к ROS-графу Linux-хоста используйте одинаковый
`ROS_DOMAIN_ID`, а также `--network host --ipc host`, например:

```bash
docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=0 \
  obstacle_detector:humble ros2 launch obstacle_detector_ros pipeline.launch.py \
  start_loader:=false input_reliability:=best_effort cloud_topic:=/lidar_points
```

Внешний `ros2 bag play` тоже можно использовать вместе с обработчиком.
Обратное подтверждение действует только между нашими загрузчиком и обработчиком.
В живом режиме очередь ограничена; при перегрузке кадры могут теряться.
Временные правила существующего трекера рассчитаны на последовательный поток 10 Гц.
Разрыв по timestamp более `max_gap_sec` (по умолчанию 0,25 с), возврат времени назад
или смена `frame_id` сбрасывают состояние. `max_gap_sec:=0.0` отключает только
проверку больших разрывов; пороги трекера не перенастраиваются автоматически.

## Выходные топики

| Топик | Тип и содержание |
|---|---|
| `/obstacle_detector/result` | `std_msgs/msg/String`: строгий JSON, один результат на кадр |
| `/obstacle_detector/corridor` | `PointCloud2`: XYZ точек выбранного коридора, включая нижний слой |
| `/obstacle_detector/obstacles` | `PointCloud2`: реально наблюдаемые XYZ подтверждённых препятствий текущего кадра |

Пустое облако препятствий само по себе не означает `CLEAR`: при прогнозируемом
препятствии текущих отражений может не быть. Статус берётся из JSON.
Все выходы имеют исходный `frame_id` и timestamp входного сообщения.

JSON содержит `schema_version`, `header: {frame_id, timestamp_ns}`, `frame_index`,
`processing_ok`, `status`, `path_clear`, `distance_m`, `confidence`, `blocking_ids`,
`tracks`, `route_available`, `route_status`, `travel_m`, `timing_ms`, `removed_points`
и `reset_reason`. При ошибке содержит `error`; поля успешного анализа могут отсутствовать.
`distance_m` — до ближайшего подтверждённого препятствия, иначе существующего
объекта с предупреждением; при отсутствии значения — `null`.

Трек содержит ID, решение, признаки подтверждения/наблюдения, тип, расстояние,
центр в координатах сенсора и оценку скорости **в метрах на кадр**.
`confidence` — эвристическая оценка, а не калиброванная вероятность.
Невалидные координаты NaN/Inf удаляются вместе с соответствующими `ring`;
полностью пустой кадр считается ошибкой. Ошибка сбрасывает трекер.

QoS облаков загрузчика — reliable / volatile; вход обработчика по умолчанию
reliable, для внешнего сенсора доступен best_effort. Выход результата —
reliable / transient_local / depth 1: поздний подписчик получает последнее решение.
Одновременно используйте один загрузчик и один обработчик на одной паре топиков.

```bash
ros2 topic echo /obstacle_detector/result --qos-durability transient_local
ros2 topic info /lidar_points --verbose
```

`output_jsonl` дополнительно сохраняет каждый результат отдельной строкой с flush
перед подтверждением загрузчику. Существующий файл не перезаписывается. Без этого
параметра результаты доступны в ROS-топиках. ROS-режим не создаёт MP4 или прежние CSV;
для них остаётся файловый запуск:

```bash
docker run --rm -v "$PWD/data:/data:ro" -v "$PWD/output:/output" \
  obstacle_detector:humble run.py /data/recording --visualize --output-dir /output/demo
```

## Установка без Docker на Ubuntu 22.04

После установки ROS 2 Humble, в workspace с этим репозиторием в `src/`:

```bash
source /opt/ros/humble/setup.bash
sudo apt-get install python3-pip python3-colcon-common-extensions \
  ros-humble-rclpy ros-humble-sensor-msgs ros-humble-std-msgs ros-humble-launch-ros
python3 -m pip install -r src/LCT-2026.NIIstovye/requirements-humble.txt
colcon build --packages-select obstacle_detector_ros
source install/setup.bash
ros2 launch obstacle_detector_ros pipeline.launch.py input:=/data/recording
```

Для Python 3.10 применяется `requirements-humble.txt`. Прежние файлы
`requirements.txt` и `requirements-visualization.txt` относятся к Windows/Python 3.11.
Параметры нод задаются при старте и доступны через `ros2 param list/describe`;
во время работы они доступны только для чтения. У обработчика также есть
`threads`, `rise_m`, `min_unique`, `self_return_m`, `queue_depth`,
`corridor_topic`, `obstacles_topic`. Загрузчик без нашего обработчика можно запустить
с `wait_for_result:=false`; в этом режиме подтверждение обработки не проверяется.

## Проверки и источники

Локальные проверки конвертации и алгоритма без ROS:

```bash
python3 -m pip install pytest==8.3.5
python3 -m pytest tests -q
```

Без `rclpy` модуль нативных интеграционных тестов пропускается. Это не подтверждение
работы DDS. Полный комплект выполняет `docker build --target test .`;
локально выполненные проверки и ограничения перечислены в [VALIDATION_ROS2.md](VALIDATION_ROS2.md).

Официальные основания: [образ ROS](https://hub.docker.com/_/ros),
[Ubuntu 22.04 для Humble](https://github.com/ros2/ros2_documentation/blob/humble/source/Installation/Ubuntu-Install-Debs.rst),
[QoS ROS 2](https://github.com/ros2/ros2_documentation/blob/humble/source/Concepts/Intermediate/About-Quality-of-Service-Settings.rst).
