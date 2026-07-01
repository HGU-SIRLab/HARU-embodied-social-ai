# robot_face/main.py
# HARU 통합 버전 — 키보드/마우스/30초 타임아웃 제거
# 감정 변경은 emotion_queue를 통해서만 (ROS2 /haru_expression Int32)

import pygame
import random
import math
import queue
import traceback
import threading

from emotions.neutral import Emotion as NeutralEmotion
from emotions.happy import Emotion as HappyEmotion
from emotions.excited import Emotion as ExcitedEmotion
from emotions.tender import Emotion as TenderEmotion
from emotions.scared import Emotion as ScaredEmotion
from emotions.angry import Emotion as AngryEmotion
from emotions.sad import Emotion as SadEmotion
from emotions.surprised import Emotion as SurprisedEmotion
from emotions.listening import Emotion as ListeningEmotion
from emotions.thinking import Emotion as ThinkingEmotion
from emotions.close import Emotion as CloseEmotion
from emotions.scanning import Emotion as ScanningEmotion
from emotions.sleepy import Emotion as SleepyEmotion
from emotions.wake import Emotion as WakeEmotion
from emotions import eyebrow
from emotions import cheeks

faceColor = (0, 0, 0)


class RobotFaceApp:
    def __init__(self, emotion_queue=None, stop_event=None, ptt_thread=None):
        pygame.init()

        monitor_sizes = pygame.display.get_desktop_sizes()
        monitor_index = 0
        if len(monitor_sizes) > 1:
            monitor_index = 1

        self.desktop_width, self.desktop_height = monitor_sizes[monitor_index]
        self.original_width, self.original_height = 800, 480
        self.scale_factor = min(
            self.desktop_width / self.original_width,
            self.desktop_height / self.original_height
        )
        # 모니터 전체를 채우도록 창 크기를 데스크톱 해상도와 동일하게 설정
        self.scaled_width = self.desktop_width
        self.scaled_height = self.desktop_height

        self.screen = pygame.display.set_mode(
            (self.scaled_width, self.scaled_height),
            pygame.FULLSCREEN | pygame.NOFRAME,
            display=monitor_index
        )
        self.base_surface = pygame.Surface((self.original_width, self.original_height))
        pygame.display.set_caption("HARU Face")
        self.clock = pygame.time.Clock()

        self.common_data = {
            'left_eye':  (self.original_width // 2 - 200, self.original_height // 2),
            'right_eye': (self.original_width // 2 + 200, self.original_height // 2),
            'offset': [0.0, 0.0],
            'time': 0,
            'scale_factor': self.scale_factor,
        }

        self.emotion_queue = emotion_queue
        self.stop_event = stop_event or threading.Event()
        self.ptt_thread = ptt_thread
        self.target_offset = [0.0, 0.0]
        self.move_speed = 1.5
        self.max_pupil_move_distance = 20
        self.is_blinking = False
        self.blink_progress = 0
        self.normal_blink_speed = 15

        pygame.time.set_timer(pygame.USEREVENT + 1, random.randint(2000, 5000))
        pygame.time.set_timer(pygame.USEREVENT + 2, random.randint(2000, 5000))

        self.emotions = {
            "NEUTRAL":   NeutralEmotion(),
            "HAPPY":     HappyEmotion(),
            "EXCITED":   ExcitedEmotion(),
            "TENDER":    TenderEmotion(),
            "SCARED":    ScaredEmotion(),
            "ANGRY":     AngryEmotion(),
            "SAD":       SadEmotion(),
            "SURPRISED": SurprisedEmotion(),
            "LISTENING": ListeningEmotion(),
            "THINKING":  ThinkingEmotion(),
            "CLOSE":     CloseEmotion(),
            "SCANNING":  ScanningEmotion(),
            "SLEEPY":    SleepyEmotion(),
            "WAKE":      WakeEmotion(),
        }
        self.current_emotion_key = "NEUTRAL"

        self.eyebrow_drawers = {
            "ANGRY":     eyebrow.draw_angry_eyebrows,
            "SAD":       eyebrow.draw_sad_eyebrows,
            "THINKING":  eyebrow.draw_thinking_eyebrows,
            "LISTENING": eyebrow.draw_thinking_eyebrows,
        }
        self.cheek_drawers = {
            "HAPPY":  cheeks.draw_happy_cheeks,
            "TENDER": cheeks.draw_tender_cheeks,
        }

    def change_emotion(self, new_emotion_key):
        if new_emotion_key not in self.emotions:
            print(f"경고: 알 수 없는 감정 키 '{new_emotion_key}'는 무시됩니다.")
            return
        if self.current_emotion_key != new_emotion_key:
            print(f"감정 변경: {self.current_emotion_key} -> {new_emotion_key}")
            self.current_emotion_key = new_emotion_key
            if hasattr(self.emotions[self.current_emotion_key], 'reset'):
                self.emotions[self.current_emotion_key].reset()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.stop_event.set()
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.stop_event.set()
                return False
            if event.type == pygame.USEREVENT + 1:
                if self.current_emotion_key not in ["LISTENING", "SCANNING"]:
                    self.target_offset = self.get_random_target_offset()
            if event.type == pygame.USEREVENT + 2 and not self.is_blinking:
                self.is_blinking = True
                self.blink_progress = 0
        return True

    def update(self):
        if self.stop_event.is_set():
            return False

        if self.emotion_queue:
            try:
                command = self.emotion_queue.get_nowait()
                self.change_emotion(command)
            except queue.Empty:
                pass

        if self.current_emotion_key == "LISTENING":
            self.target_offset = [0.0, 0.0]
            self.common_data['offset'] = [0.0, 0.0]
        elif self.current_emotion_key == "SCANNING":
            current_time = pygame.time.get_ticks() / 1000.0
            rotation_radius = 40
            rotation_speed = 2.0
            offset_x = math.cos(current_time * rotation_speed) * rotation_radius
            offset_y = math.sin(current_time * rotation_speed) * rotation_radius
            self.common_data['offset'][0] = offset_x
            self.common_data['offset'][1] = offset_y
        else:
            dx = self.target_offset[0] - self.common_data['offset'][0]
            dy = self.target_offset[1] - self.common_data['offset'][1]
            dist = math.hypot(dx, dy)
            if dist > self.move_speed:
                self.common_data['offset'][0] += (dx / dist) * self.move_speed
                self.common_data['offset'][1] += (dy / dist) * self.move_speed

        if self.is_blinking:
            self.blink_progress += self.normal_blink_speed
            if self.blink_progress >= 200:
                self.is_blinking = False

        self.common_data['time'] = pygame.time.get_ticks()
        return True

    def draw(self):
        self.screen.fill((0, 0, 0))
        self.base_surface.fill(faceColor)
        self.emotions[self.current_emotion_key].draw(self.base_surface, self.common_data)

        if self.is_blinking and self.current_emotion_key != "SCANNING":
            progress = self.blink_progress if self.blink_progress <= 100 else 200 - self.blink_progress
            for eye_center in [self.common_data['left_eye'], self.common_data['right_eye']]:
                pygame.draw.rect(self.base_surface, faceColor,
                                 (eye_center[0] - 100, eye_center[1] - 150, 200, progress + 50))
                pygame.draw.rect(self.base_surface, faceColor,
                                 (eye_center[0] - 100, eye_center[1] + 100 - progress, 200, progress + 50))

        if self.current_emotion_key in self.eyebrow_drawers:
            self.eyebrow_drawers[self.current_emotion_key](self.base_surface, self.common_data)
        if self.current_emotion_key in self.cheek_drawers:
            self.cheek_drawers[self.current_emotion_key](self.base_surface, self.common_data)

        scaled = pygame.transform.scale(self.base_surface, (self.scaled_width, self.scaled_height))
        self.screen.blit(scaled, (0, 0))
        pygame.display.flip()

    def get_random_target_offset(self):
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(0, self.max_pupil_move_distance)
        return [math.cos(angle) * distance, math.sin(angle) * distance]

    def run(self):
        self.change_emotion("NEUTRAL")
        while not self.stop_event.is_set():
            try:
                if not self.handle_events():
                    break
                if not self.update():
                    break
                self.draw()
                self.clock.tick(60)
            except Exception as e:
                print(f"Face App 오류: {type(e).__name__} - {e}")
                traceback.print_exc()
                break

        if self.ptt_thread and self.ptt_thread.is_alive():
            self.ptt_thread.join()
        pygame.quit()


def run_face_app(emotion_q, stop_event, ptt_thread: threading.Thread):
    try:
        app = RobotFaceApp(
            emotion_queue=emotion_q,
            stop_event=stop_event,
            ptt_thread=ptt_thread,
        )
        app.run()
    except Exception as e:
        print(f"Face App 시작 오류: {e}")
        traceback.print_exc()
