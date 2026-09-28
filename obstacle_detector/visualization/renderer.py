"""Draw the result of a processed frame without running inference again."""
from . import layout as ui
from .objects import gauge_display
from .points import CAMERA, STYLE, annotate, render


class FrameVisualizer:
    """Keep display box history; render() returns a Pillow RGB image."""

    def __init__(self, *, fps=10., bev_limit=None, font=None, font_bold=None):
        ui.configure_fonts(font, font_bold)
        self.fps = fps
        self.bev_limit = bev_limit
        self.history = {}

    def reset(self):
        self.history.clear()

    def prepare(self, xyz, result, *, frame_idx=0, frames_total=1,
                recording_label='', timestamp_ns=None, rotation=None):
        shown = gauge_display(result, xyz, self.history, rotation)
        estimated = bool(shown.obstacles and shown.obstacles[0].predicted)
        subtitle = 'Положение препятствия прогнозируется' if estimated else None
        if result.status == 'CAUTION':
            entering = any(t.decision == 'CAUTION' and t.relation == 'inside' and t.seen for t in result.tracks)
            subtitle = ('Коридор недоступен для проверки' if result.geometry is None or result.detection is None else
                        'Положение объекта прогнозируется' if estimated else
                        'Подтверждается попадание в габарит' if entering else
                        'Объект в пограничной зоне' if shown.obstacles else
                        'Ожидается подтверждение объекта')
        return dict(xyz=xyz, analysis=shown, recording_label=recording_label,
            frame_idx=frame_idx, timestamp_ns=timestamp_ns, frames_total=frames_total,
            time_s=frame_idx / self.fps, duration_s=frames_total / self.fps,
            decision=dict(status_out=result.status, distance_temporal_input_m=result.distance_m,
                          distance_estimated=estimated),
            subtitle=subtitle, bev_limit=self.bev_limit)

    @staticmethod
    def render_payload(payload):
        shown = payload['analysis']

        def renderer(points, mask, width, height):
            picture, project, stats = render(points, mask, STYLE, CAMERA, warning_mask=shown.warning_mask)
            return annotate(picture, project, shown), project, stats['drawn_returns']

        image, _ = ui.main(payload, renderer=renderer,
            left_title='Облако точек', left_subtitle=CAMERA['title'],
            bev_forward_limit=payload['bev_limit'],
            badge_label='obstacle_detector',
            algorithm_name='obstacle_detector', status_subtitle=payload['subtitle'],
            annotate_perspective=False, save=False, quiet=True)
        return image

    def render(self, xyz, result, **kwargs):
        return self.render_payload(self.prepare(xyz, result, **kwargs))
