"""
HARU 표정 디스플레이 노드
/haru_expression (Int32) → pygame 얼굴 애니메이션
표정 ID: 0=neutral 1=joy 2=sadness 3=curiosity 4=surprise 5=empathy 6=thinking 7=concern
"""
import math
import os
import threading
import time

import pygame
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

# ── 표정별 설정 ────────────────────────────────────────────────────────────────
EXPRESSIONS = {
    #  id : (이름,      배경색,          눈 파라미터,    입 파라미터)
    0:  ('neutral',   (40,  40,  60),   'normal',      'flat'),
    1:  ('joy',       (60,  90,  30),   'happy',       'bigsmile'),
    2:  ('sadness',   (30,  30,  80),   'sad',         'frown'),
    3:  ('curiosity', (80,  60,  20),   'curious',     'slight_smile'),
    4:  ('surprise',  (70,  50,  90),   'wide',        'open'),
    5:  ('empathy',   (70,  30,  50),   'soft',        'gentle'),
    6:  ('thinking',  (20,  60,  80),   'look_up',     'flat'),
    7:  ('concern',   (80,  40,  20),   'worried',     'frown_slight'),
}

DISPLAY_W = int(os.environ.get('HARU_DISP_W', '800'))
DISPLAY_H = int(os.environ.get('HARU_DISP_H', '600'))
FPS       = 30
BLINK_INTERVAL = 4.0   # 눈 깜빡임 주기 (초)
BLINK_DUR      = 0.12  # 깜빡임 지속 시간 (초)


def lerp(a, b, t):
    return a + (b - a) * t


class FaceRenderer:
    """간결한 원형 HARU 얼굴을 pygame Surface에 그린다."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.cx, self.cy = w // 2, h // 2
        self.face_r = min(w, h) // 2 - 30

        # 애니메이션 상태
        self.current_id   = 0
        self.target_id    = 0
        self.anim_t       = 1.0   # 1.0 = 전환 완료
        self.ANIM_SPEED   = 3.0   # 초당 진행율
        self._blink_t     = time.monotonic()
        self._blinking    = False

    def set_expression(self, expr_id: int):
        if expr_id not in EXPRESSIONS:
            return
        if expr_id == self.target_id:
            return
        self.current_id = self.target_id
        self.target_id  = expr_id
        self.anim_t     = 0.0

    def update(self, dt: float):
        if self.anim_t < 1.0:
            self.anim_t = min(1.0, self.anim_t + dt * self.ANIM_SPEED)

        now = time.monotonic()
        if not self._blinking and now - self._blink_t > BLINK_INTERVAL:
            self._blinking = True
            self._blink_t  = now
        if self._blinking and now - self._blink_t > BLINK_DUR:
            self._blinking = False
            self._blink_t  = now

    def draw(self, surface: pygame.Surface):
        t = self._ease(self.anim_t)

        # 배경색 보간
        bg_cur = EXPRESSIONS[self.current_id][1]
        bg_tgt = EXPRESSIONS[self.target_id][1]
        bg = tuple(int(lerp(a, b, t)) for a, b in zip(bg_cur, bg_tgt))
        surface.fill(bg)

        # 표정 파라미터 (전환 중에는 target 우선)
        eid = self.target_id if t > 0.5 else self.current_id
        _, _, eye_style, mouth_style = EXPRESSIONS[eid]

        self._draw_face(surface, eye_style, mouth_style, t)
        self._draw_label(surface, EXPRESSIONS[self.target_id][0])

    def _ease(self, t):
        """ease-in-out"""
        return t * t * (3 - 2 * t)

    def _draw_face(self, surface, eye_style, mouth_style, anim_t):
        cx, cy = self.cx, self.cy
        r = self.face_r

        # 얼굴 외곽 (흰색 원)
        face_color = (245, 230, 210)
        pygame.draw.circle(surface, face_color, (cx, cy), r)
        pygame.draw.circle(surface, (200, 180, 160), (cx, cy), r, 3)

        # 눈 위치
        eye_y_off = -r // 6
        eye_x_off = r // 4

        self._draw_eyes(surface, cx, cy, r, eye_x_off, eye_y_off, eye_style)
        self._draw_mouth(surface, cx, cy, r, mouth_style)

        # 볼 (joy/empathy일 때)
        if eye_style in ('happy', 'soft'):
            cheek_color = (255, 180, 180, 100)
            cheek_surf = pygame.Surface((r // 2, r // 4), pygame.SRCALPHA)
            cheek_surf.fill((255, 160, 160, 80))
            surface.blit(cheek_surf, (cx - r // 2 - r // 8, cy + r // 6))
            surface.blit(cheek_surf, (cx + r // 8, cy + r // 6))

    def _draw_eyes(self, surface, cx, cy, r, ex, ey, style):
        ew = r // 5   # 눈 가로 반경
        eh = r // 6   # 눈 세로 반경 (기본)
        eye_color = (30, 30, 60)
        gleam = (255, 255, 255)

        positions = [(cx - ex, cy + ey), (cx + ex, cy + ey)]

        # 눈 스타일별 파라미터
        style_map = {
            # style : (eh_mult, shape, brow_offset)
            'normal':    (1.0, 'ellipse',   -eh - 4),
            'happy':     (0.4, 'arc_happy',  -eh - 4),
            'sad':       (1.0, 'ellipse',    eh),
            'curious':   (1.3, 'ellipse',   -eh - 8),
            'wide':      (1.6, 'ellipse',   -eh - 10),
            'soft':      (0.7, 'ellipse',   -eh - 2),
            'look_up':   (1.0, 'look_up',   -eh - 6),
            'worried':   (1.0, 'ellipse',    eh // 2),
        }
        eh_mult, shape, brow_off = style_map.get(style, (1.0, 'ellipse', -eh - 4))
        eh_actual = max(3, int(eh * eh_mult))

        for (px, py) in positions:
            if self._blinking or shape == 'arc_happy':
                # 눈 감기 / 초승달 눈
                points = []
                for a in range(0, 181, 10):
                    rad = math.radians(a)
                    sign = -1 if shape == 'arc_happy' else 1
                    x = px + ew * math.cos(math.radians(180 - a))
                    y = py + sign * eh_actual * math.sin(rad)
                    points.append((x, y))
                if len(points) >= 3:
                    pygame.draw.lines(surface, eye_color, False, points, 3)
            else:
                pygame.draw.ellipse(surface, eye_color,
                                    (px - ew, py - eh_actual, ew * 2, eh_actual * 2))
                # 눈 반짝임
                pygame.draw.circle(surface, gleam,
                                   (px - ew // 3, py - eh_actual // 3), max(2, ew // 5))

            # 눈썹
            self._draw_brow(surface, px, py, ew, brow_off, style, eye_color, cx)

        # look_up: 눈동자 위로
        if style == 'look_up':
            for (px, py) in positions:
                pygame.draw.ellipse(surface, eye_color,
                                    (px - ew, py - eh_actual, ew * 2, eh_actual * 2))
                # 눈동자를 위로
                pygame.draw.circle(surface, (60, 60, 100),
                                   (px, py - eh_actual // 2), ew // 2)

    def _draw_brow(self, surface, px, py, ew, off, style, color, cx):
        bx1 = px - ew
        bx2 = px + ew
        by  = py + off
        # worried/sad: 안쪽이 올라가는 찡그린 눈썹
        if style in ('worried', 'sad'):
            inner_rise = 6 if style == 'worried' else 4
            by1 = by + (inner_rise if px < cx else -inner_rise)
            by2 = by + (-inner_rise if px < cx else inner_rise)
        else:
            by1, by2 = by, by
        pygame.draw.line(surface, color, (bx1, by1), (bx2, by2), 3)

    def _draw_mouth(self, surface, cx, cy, r, style):
        mouth_y = cy + r // 3
        mw = r // 3    # 입 가로 반경
        color = (80, 40, 40)

        style_map = {
            'flat':         (0,    0),     # 직선
            'bigsmile':     (mw // 2, 1),  # 큰 미소
            'slight_smile': (mw // 4, 1),  # 약한 미소
            'gentle':       (mw // 3, 1),  # 부드러운 미소
            'frown':        (mw // 2, -1), # 찡그림
            'frown_slight': (mw // 4, -1), # 약한 찡그림
            'open':         (0,    2),     # 벌린 입 (동그라미)
        }
        depth, sign = style_map.get(style, (0, 0))

        if style == 'open':
            # 놀람: 타원형 입
            pygame.draw.ellipse(surface, color,
                                (cx - mw // 2, mouth_y - mw // 3, mw, mw * 2 // 3), 3)
        elif depth == 0:
            # 직선
            pygame.draw.line(surface, color, (cx - mw, mouth_y), (cx + mw, mouth_y), 3)
        else:
            # 베지어 곡선 근사 (포물선)
            points = []
            for i in range(21):
                t = i / 20
                x = cx - mw + 2 * mw * t
                y = mouth_y + sign * depth * 4 * t * (1 - t)
                points.append((int(x), int(y)))
            if len(points) >= 2:
                pygame.draw.lines(surface, color, False, points, 3)

    def _draw_label(self, surface, name: str):
        if not pygame.font.get_init():
            return
        font = pygame.font.SysFont('DejaVuSans', 24)
        txt = font.render(name, True, (200, 200, 220))
        surface.blit(txt, (10, 10))


class HaruExpressionNode(Node):
    def __init__(self):
        super().__init__('haru_expression_node')

        self._sub = self.create_subscription(
            Int32, 'haru_expression', self._on_expression, 10)

        self._current_expr = 0
        self._lock = threading.Lock()

        self.get_logger().info('Expression Node 시작 — pygame 표정 디스플레이')

    def _on_expression(self, msg: Int32):
        with self._lock:
            self._current_expr = msg.data
        self.get_logger().info(f'[Expr] {msg.data} → {EXPRESSIONS.get(msg.data, ("?",))[0]}')

    def get_current(self) -> int:
        with self._lock:
            return self._current_expr


def run_display(node: HaruExpressionNode):
    """pygame 메인 루프 (별도 스레드 불가 → main에서 직접 호출)."""
    os.environ.setdefault('DISPLAY', ':0')

    pygame.init()
    pygame.font.init()
    try:
        screen = pygame.display.set_mode(
            (DISPLAY_W, DISPLAY_H),
            pygame.NOFRAME | pygame.SHOWN
        )
    except pygame.error as e:
        node.get_logger().error(f'pygame 디스플레이 오류: {e} — 헤드리스 모드')
        return

    pygame.display.set_caption('HARU')
    clock = pygame.time.Clock()
    renderer = FaceRenderer(DISPLAY_W, DISPLAY_H)
    prev_time = time.monotonic()

    node.get_logger().info(
        f'표정 디스플레이 시작 ({DISPLAY_W}×{DISPLAY_H}) — DISPLAY={os.environ.get("DISPLAY")}')

    while rclpy.ok():
        # pygame 이벤트
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return

        # 표정 업데이트
        expr_id = node.get_current()
        renderer.set_expression(expr_id)

        now = time.monotonic()
        dt = now - prev_time
        prev_time = now
        renderer.update(dt)
        renderer.draw(screen)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


def main(args=None):
    rclpy.init(args=args)
    node = HaruExpressionNode()

    # ROS2 spin을 별도 스레드에서 실행 (pygame은 main thread 필요)
    spin_thread = threading.Thread(
        target=lambda: rclpy.spin(node), daemon=True)
    spin_thread.start()

    try:
        run_display(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == '__main__':
    main()
