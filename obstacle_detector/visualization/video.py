"""Bounded parallel rendering and ordered H.264 encoding."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import tempfile

import imageio_ffmpeg
from PIL import Image

from . import layout as ui
from .renderer import FrameVisualizer


def _initialize_worker(font, font_bold):
    ui.configure_fonts(font, font_bold)


def _render(payload):
    return FrameVisualizer.render_payload(payload).tobytes()


class VideoWriter:
    def __init__(self, path, *, fps=10., workers=3, font=None, font_bold=None):
        self.path = Path(path)
        self.partial = self.path.with_name(self.path.stem + '.rendering.mp4')
        self.fps, self.workers = fps, workers
        self.font, self.font_bold = font, font_bold
        self.pending = deque()
        self.frames = 0
        self.preview = None
        self.preview_rank = -1

    def __enter__(self):
        self.environment = {key: os.environ.get(key) for key in
                            ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}
        # Inference has already loaded its numerical libraries. These settings
        # apply to the newly spawned rendering processes only.
        for key in self.environment:
            os.environ[key] = '1'
        self.log = tempfile.TemporaryFile()
        command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{ui.W}x{ui.H}',
            '-framerate', str(self.fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
            '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p', '-threads', '2',
            '-movflags', '+faststart', str(self.partial)]
        try:
            # ИСПРАВЛЕНИЕ 1: close_fds=True предотвращает наследование дескрипторов самим ffmpeg
            self.encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=self.log,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), close_fds=True)

            # ИСПРАВЛЕНИЕ 2: Явно используем 'spawn', чтобы воркеры не наследовали дескрипторы пайпа
            mp_context = mp.get_context('spawn')
            self.pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=mp_context,
                                            initializer=_initialize_worker,
                                            initargs=(self.font, self.font_bold))
        except BaseException:
            if hasattr(self, 'encoder'):
                self.encoder.stdin.close()
                self.encoder.wait()
            self.log.close()
            self._restore_environment()
            raise
        return self

    def add(self, payload):
        rank = {'CLEAR': 0, 'CAUTION': 1, 'BLOCKED': 2}[payload['decision']['status_out']]
        self.pending.append((self.pool.submit(_render, payload), rank))
        if len(self.pending) >= 2 * self.workers:
            self._write_next()

    def _write_next(self):
        future, rank = self.pending.popleft()
        pixels = future.result()
        self.encoder.stdin.write(pixels)
        self.frames += 1
        if rank > self.preview_rank:
            self.preview, self.preview_rank = pixels, rank

    def _restore_environment(self):
        for key, value in self.environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                while self.pending:
                    self._write_next()
                self.encoder.stdin.close()
                code = self.encoder.wait(timeout=60)
                if code != 0:
                    self.log.seek(0)
                    raise RuntimeError('FFmpeg: ' + self.log.read().decode('utf-8', errors='replace'))
                self.partial.replace(self.path)
                if self.preview is not None:
                    Image.frombytes('RGB', (ui.W, ui.H), self.preview).save(self.path.parent / 'preview.png')
        finally:
            for future, _ in self.pending:
                future.cancel()
            self.pool.shutdown(wait=True, cancel_futures=True)
            if not self.encoder.stdin.closed:
                self.encoder.stdin.close()
            if self.encoder.poll() is None:
                try:
                    self.encoder.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.encoder.terminate()
                    self.encoder.wait()
            self.log.close()
            self._restore_environment()