"""Process a cloud or a recording, with optional rendering."""
import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from . import Config, GaugeConfig, StreamProcessor, __version__
from .io import FrameSource


CSV_FIELDS = ('frame_idx', 'timestamp_ns', 'status', 'path_clear', 'distance_m',
              'confidence', 'blocking_ids', 'caution_ids', 'travel_m',
              'route_available', 'elapsed_ms')


def process_recording(source_path, output_dir, *, topic=None, visualize=False,
                      max_frames=0, config=None, gauge_config=None, fps=10.,
                      workers=3, font=None, font_bold=None):
    """Run frames in order using one processor; return the saved summary.

    Optional rendering receives completed results and never changes inference.
    Frames and render jobs use bounded memory; full clouds are not cached on disk.
    """
    print("Внимание: начальная инициализация может занять время (компиляция Numba, загрузка моделей и т.д.).", flush=True)

    source_path, output_dir = Path(source_path).resolve(), Path(output_dir).resolve()
    if max_frames < 0 or not np.isfinite(fps) or fps <= 0 or not 1 <= workers <= 8:
        raise ValueError('Use max_frames >= 0, fps > 0 and workers in 1..8')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Choose an empty output directory: {output_dir}')

    processor = StreamProcessor(config, gauge_config)
    started = time.perf_counter()
    counts = Counter()
    first_blocked = None

    with ExitStack() as stack:
        source = stack.enter_context(FrameSource(source_path, topic=topic))
        count = min(len(source), max_frames) if max_frames else len(source)
        if count == 0:
            raise ValueError('The input contains no frames')

        visualizer = video = None
        if visualize:
            from .visualization import FrameVisualizer
            from .visualization.video import VideoWriter
            from .visualization.layout import bev_axis
            visualizer = FrameVisualizer(fps=fps, font=font, font_bold=font_bold)
            limit = 0.
            last_progress = time.monotonic()
            print(f'BEV extent: scanning {count} frames', flush=True)
            for index in range(count):
                limit = max(limit, bev_axis(source.read(index).xyz)[2])
                if time.monotonic() - last_progress >= 15 or index + 1 == count:
                    print(f'BEV extent: {index + 1}/{count}', flush=True)
                    last_progress = time.monotonic()
            visualizer.bev_limit = limit

        output_dir.mkdir(parents=True, exist_ok=True)
        if visualize and source.single is None:
            video = stack.enter_context(VideoWriter(output_dir / 'video.mp4', fps=fps,
                                                     workers=workers, font=font, font_bold=font_bold))
        csv_file = stack.enter_context((output_dir / 'frames.csv').open('x', newline='', encoding='utf-8-sig'))
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()

        last_progress = time.monotonic()
        for index in range(count):
            frame = source.read(index)
            xyz = frame.xyz.astype(np.float64, copy=False)
            result = processor.process(xyz, frame.ring, frame.timestamp_ns)

            # === ОБНОВЛЕННЫЙ ВЫВОД ===
            if index == 0:
                time_to_first = time.perf_counter() - started
                print(f"Первый кадр получен через {time_to_first:.2f} с (включая инициализацию)", flush=True)
            else:
                process_time_ms = result.timing_ms.get('total', 0.0)
                # end='\r' перезаписывает текущую строку, создавая эффект обновления прогресса
                print(f"Кадр {index + 1}/{count} | {result.status} | Обработка: {process_time_ms:.2f} мс", flush=True, end='\r')
            # =========================

            writer.writerow(dict(frame_idx=index, timestamp_ns=frame.timestamp_ns,
                status=result.status, path_clear=result.path_clear, distance_m=result.distance_m,
                confidence=result.confidence, blocking_ids=';'.join(str(t.id) for t in result.blocking),
                caution_ids=';'.join(str(t.id) for t in result.tracks if t.decision == 'CAUTION'),
                travel_m=result.travel_m, route_available=result.geometry is not None,
                elapsed_ms=result.timing_ms.get('total', 0.0)))

            counts[result.status] += 1
            if result.status == 'BLOCKED' and first_blocked is None:
                first_blocked = index

            if visualizer is not None:
                payload = visualizer.prepare(xyz, result, frame_idx=index, frames_total=count,
                    recording_label=source_path.stem, timestamp_ns=frame.timestamp_ns,
                    rotation=processor.last_motion.rotation)
                if video is None:
                    visualizer.render_payload(payload).save(output_dir / 'frame.png')
                else:
                    video.add(payload)

            # Оставили только flush для безопасности записи на диск, убрали лишний print
            if time.monotonic() - last_progress >= 15 or index + 1 == count:
                csv_file.flush()
                last_progress = time.monotonic()

        # Перенос строки после завершения цикла, чтобы курсор перешел на новую строку после последнего '\r'
        print(flush=True)

    summary = dict(version=__version__, input=str(source_path), topic=source.topic,
        frames=count, status_counts={s: counts[s] for s in ('BLOCKED', 'CAUTION', 'CLEAR')},
        first_blocked_frame=first_blocked, elapsed_s=time.perf_counter() - started,
        config=asdict(processor.config), gauge_config=asdict(processor.gauge_decision.config),
        visualization=('video.mp4' if video is not None else 'frame.png') if visualize else None,
        video_fps=fps if video is not None else None, complete=True)

    (output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Complete: {output_dir} | {dict(counts)}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path, help='DB3, bag directory, NPY or NPZ; one or more independent inputs')
    parser.add_argument('--output-dir', type=Path, default=Path('result'))
    parser.add_argument('--topic', help='PointCloud2 topic; auto-selected when there is exactly one')
    parser.add_argument('--visualize', action='store_true', help='Also save cloud + BEV video (PNG for NPY/NPZ)')
    parser.add_argument('--threads', type=int, default=3, help='Numba threads for inference (default: 3)')
    parser.add_argument('--workers', type=int, default=3, help='Rendering processes, 1..8 (default: 3)')
    parser.add_argument('--fps', type=float, default=10., help='Playback FPS; does not change inference')
    parser.add_argument('--max-frames', type=int, default=0, help='First N frames; 0 = entire recording')
    parser.add_argument('--rise-m', type=float, default=.10, help='Height threshold above the estimated track surface')
    parser.add_argument('--min-unique', type=int, default=3, help='Minimum distinct returns in a group')
    parser.add_argument('--self-return-m', type=float, default=2.5, help='Ignore sensor mounting returns inside this range')
    parser.add_argument('--font', type=Path, help='Optional regular TTF with Cyrillic support')
    parser.add_argument('--font-bold', type=Path, help='Optional bold TTF; specify together with --font')
    parser.add_argument('--version', action='version', version=f'obstacle_detector {__version__}')
    args = parser.parse_args()

    if args.max_frames < 0 or not np.isfinite(args.fps) or args.fps <= 0 or not 1 <= args.workers <= 8:
        parser.error('Use --max-frames >= 0, --fps > 0 and --workers in 1..8')
    if (args.font is None) != (args.font_bold is None):
        parser.error('Specify both --font and --font-bold')

    try:
        config = Config(threads=args.threads, rise_m=args.rise_m,
                        min_unique=args.min_unique, self_return_m=args.self_return_m)
        names = [p.stem for p in args.inputs]
        if len(set(names)) != len(names):
            parser.error('Independent inputs must have different names')
        for path in args.inputs:
            if not path.exists():
                parser.error(f'Input does not exist: {path}')
        for path in args.inputs:
            output = args.output_dir / path.stem if len(args.inputs) > 1 else args.output_dir
            process_recording(path, output, topic=args.topic, visualize=args.visualize,
                config=config, max_frames=args.max_frames, fps=args.fps, workers=args.workers,
                font=args.font, font_bold=args.font_bold)
    except (ValueError, FileNotFoundError, FileExistsError, ModuleNotFoundError) as exc:
        parser.exit(1, f'{exc}\n')
    return 0