"""
HARU HITL (Human-in-the-Loop) 에피소드 로거 노드

흐름:
  brain → /haru_vla_raw → [이 노드] → /haru_system1_command → action_node
                               ↓
                          인간 검토/교정
                               ↓
                      data/episodes/ 저장

터미널 조작:
  [N] + Enter  : 새 에피소드 시작
  [A] + Enter  : 승인  — VLA 행동 그대로 로봇에 전송 & 로깅
  [C] + Enter  : 교정  — [D] 직접 시연 또는 [M] 수동 입력 선택
  [S] + Enter  : 스킵  — 이 스텝 로깅 안 함 (로봇은 VLA 행동 그대로 실행)
  [E] + Enter  : 에피소드 종료 & 저장
  [Q] + Enter  : 에피소드 취소 (저장 안 함)

교정 방법:
  [D] 직접 시연 (키네스테틱 티칭)
      action_node에 토크 OFF 요청 → 손으로 자세 조작 → Enter → 인코더 값 자동 캡처
  [M] 수동 입력
      관절 값을 직접 숫자로 입력 (Enter = 현재값 유지)
"""

import select
import sys
import json
import queue
import threading
import time
import io
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Int32, Bool, Float32MultiArray
from sensor_msgs.msg import CompressedImage
from PIL import Image

from .episode_writer import EpisodeWriter, EXPRESSION_LABELS, HARU_DOF_RANGES

DATA_DIR = '/home/herobot/robot_brain_workspace/data/episodes'
LINE = '━' * 56

# action_node의 HARU_LIMITS와 동일한 순서 (Float32MultiArray 인덱스 매핑)
POSITION_JOINT_ORDER = [
    'r_arm_pitch', 'l_arm_pitch', 'r_shoulder_roll', 'r_elbow_pitch',
    'l_shoulder_roll', 'l_elbow_pitch', 'head_pan', 'head_tilt', 'head_roll',
]


class HaruHITLNode(Node):
    def __init__(self):
        super().__init__('haru_hitl_node')

        # ── Subscribers ──────────────────────────────────────────
        self.create_subscription(CompressedImage, 'haru_vision/compressed',
                                 self._image_cb, 10)
        self.create_subscription(String, 'haru_vla_raw',
                                 self._vla_cb, 10)

        # 관절 상태: 키네스테틱 캡처 중 spin이 블록되어도 받을 수 있도록
        # ReentrantCallbackGroup 사용 (MultiThreadedExecutor 필요)
        _joints_group = ReentrantCallbackGroup()
        self.create_subscription(
            Float32MultiArray, 'haru_joints/state',
            self._joints_cb, 10,
            callback_group=_joints_group,
        )

        # ── Publishers ───────────────────────────────────────────
        self.pub_cmd    = self.create_publisher(String, 'haru_system1_command', 10)
        self.pub_expr   = self.create_publisher(Int32,  'haru_expression',      10)
        self.pub_torque = self.create_publisher(Bool,   'haru_joints/torque',   10)

        # ── State ────────────────────────────────────────────────
        self._latest_image: np.ndarray | None = None
        self._pending: dict | None = None
        self._state = 'IDLE'
        self._input_q: queue.Queue = queue.Queue()
        self._writer = EpisodeWriter(DATA_DIR)
        self._in_episode = False

        # 관절 상태 (키네스테틱 티칭용)
        self._latest_joints: dict | None = None
        self._latest_joints_time: float = 0.0
        self._joints_lock = threading.Lock()

        # 교정 중 _kb_worker가 stdin을 소비하지 않도록 하는 플래그
        self._in_correction = False

        # ── 키보드 입력 스레드 ────────────────────────────────────
        threading.Thread(target=self._kb_worker, daemon=True).start()

        # ── 10Hz 처리 루프 ────────────────────────────────────────
        self.create_timer(0.1, self._process)

        self._print_header()

    # ── 출력 헬퍼 ────────────────────────────────────────────────

    def _print_header(self):
        print(f'\n{LINE}')
        print(' HARU HITL 에피소드 로거 시작')
        print(f'{LINE}')
        print(' [N] 새 에피소드 시작   [Q] 프로그램 종료')
        print(f'{LINE}\n')
        sys.stdout.flush()

    def _print_vla_output(self, cmd: dict):
        action = cmd.get('action', {})
        expr_id = int(cmd.get('expression_id', 0))
        source  = cmd.get('attention_source', '-')
        context = cmd.get('attention_context', '')
        print(f'\n{LINE}')
        print(f' [에피소드 {self._writer.episode_id} | 스텝 {self._writer.current_step}] VLA 제안 [상황: {source}]')
        print(LINE)
        if context:
            print(f'  컨텍스트: {context[:60]}')
        print(f'  표정  : {EXPRESSION_LABELS.get(expr_id, "?")} ({expr_id})')
        print(f'  감정  : {cmd.get("emotion", "-")}')
        print(f'  머리  : tilt={action.get("head_tilt","-")}  '
              f'pan={action.get("head_pan","-")}  roll={action.get("head_roll","-")}')
        print(f'  오른팔: {action.get("r_arm_pitch", "-")}')
        print(f'  바퀴  : right={action.get("right_wheel",0):.0f}  left={action.get("left_wheel",0):.0f}')
        speech = cmd.get('speech', '')
        if not speech:
            print('  발화  : (침묵 — 몸짓만)')
        else:
            print(f'  발화  : {speech[:50]}')
        print(LINE)
        print(' [A] 승인   [C] 교정   [S] 스킵   [E] 에피소드 종료')
        print(f'{LINE}')
        sys.stdout.flush()

    # ── 콜백 ─────────────────────────────────────────────────────

    def _image_cb(self, msg: CompressedImage):
        try:
            pil = Image.open(io.BytesIO(bytes(msg.data))).convert('RGB')
            self._latest_image = np.array(pil, dtype=np.uint8)
        except Exception:
            pass

    def _vla_cb(self, msg: String):
        if not self._in_episode:
            return
        if self._state != 'IDLE':
            # 검토 중 도착한 brain 출력은 표시할 수 없으므로 드롭
            try:
                dropped = json.loads(msg.data)
                src = dropped.get('attention_source', '?')
            except Exception:
                src = '?'
            self.get_logger().warn(
                f'[HITL] brain 출력 드롭 (현재 state={self._state}, 트리거 출처={src}) '
                f'— 검토가 끝난 뒤 다음 이벤트를 기다립니다.'
            )
            return
        try:
            self._pending = json.loads(msg.data)
            self._state = 'WAITING'
            self._print_vla_output(self._pending)
        except Exception as e:
            self.get_logger().warn(f'VLA 명령 파싱 실패: {e}')

    def _joints_cb(self, msg: Float32MultiArray):
        if len(msg.data) < len(POSITION_JOINT_ORDER):
            return
        with self._joints_lock:
            self._latest_joints = {
                name: float(msg.data[i])
                for i, name in enumerate(POSITION_JOINT_ORDER)
            }
            self._latest_joints_time = time.time()

    # ── 키보드 입력 스레드 ────────────────────────────────────────

    def _kb_worker(self):
        """select() 기반 논블로킹 키 수신 — 교정 중에는 stdin을 소비하지 않음."""
        while True:
            if self._in_correction:
                time.sleep(0.05)
                continue
            try:
                r, _, _ = select.select([sys.stdin], [], [], 0.2)
            except Exception:
                break
            if not r:
                continue
            # select 반환 후 교정 모드 진입 여부 재확인
            if self._in_correction:
                # stdin에 데이터 있지만 _run_correction이 읽도록 넘김
                time.sleep(0.05)
                continue
            try:
                line = sys.stdin.readline().strip().upper()
            except EOFError:
                break
            if line:
                self._input_q.put(line)

    # ── 메인 처리 루프 (10Hz) ─────────────────────────────────────

    def _process(self):
        if self._input_q.empty():
            return
        key = self._input_q.get_nowait()

        if not self._in_episode:
            if key == 'N':
                path = self._writer.start_episode()
                self._in_episode = True
                self._state = 'IDLE'
                print(f'\n[START] 에피소드 시작: {path}')
                print(' brain_node의 /haru_vla_raw 토픽을 기다리는 중...\n')
                sys.stdout.flush()
            elif key == 'Q':
                print('\n[EXIT] 종료합니다.')
                sys.stdout.flush()
                rclpy.shutdown()
            return

        if self._state == 'WAITING':
            if key == 'A':
                self._accept()
            elif key == 'C':
                self._state = 'CORRECTING'
                self._in_correction = True
                threading.Thread(target=self._run_correction, daemon=True).start()
            elif key == 'S':
                self._skip()
            elif key == 'E':
                self._end_episode(accepted=True)
            elif key == 'Q':
                self._end_episode(accepted=False)

        elif self._state == 'IDLE':
            if key == 'E':
                self._end_episode(accepted=True)
            elif key == 'Q':
                self._end_episode(accepted=False)

    # ── HITL 행동 ─────────────────────────────────────────────────

    def _accept(self):
        cmd = self._pending
        self._publish_command(cmd)
        self._log_step(cmd, cmd, is_corrected=False)
        self._pending = None
        self._state = 'IDLE'
        print(f'\n[OK] 승인 — 스텝 {self._writer.current_step - 1} 저장됨\n')
        sys.stdout.flush()

    def _skip(self):
        cmd = self._pending
        self._publish_command(cmd)
        self._pending = None
        self._state = 'IDLE'
        print('\n[SKIP] 이 스텝은 로깅하지 않습니다.\n')
        sys.stdout.flush()

    def _run_correction(self):
        try:
            self._run_correction_inner()
        finally:
            self._in_correction = False

    def _run_correction_inner(self):
        print(f'\n{LINE}')
        print(' 교정 모드')
        print(LINE)
        print('  교정 방법:')
        print('    [D] 직접 시연  — 토크 OFF 후 손으로 자세 조작 (숫자 불필요)')
        print('    [M] 수동 입력  — 관절 값 직접 입력 (Enter = 현재값 유지)')
        method = input('  선택 [D/M]: ').strip().upper()
        if method not in ('D', 'M'):
            method = 'M'

        cmd    = self._pending
        action = cmd.get('action', {})
        corrected_action = dict(action)

        # ── 표정 (항상 수동: 물리 시연 불가) ──────────────────────
        expr_id = int(cmd.get('expression_id', 0))
        labels  = '  '.join(f'{k}={v}' for k, v in EXPRESSION_LABELS.items())
        print(f'\n  표정 ID ({labels})')
        val = self._prompt(f'  현재={expr_id}', expr_id)
        corrected_expr_id = int(np.clip(int(val), 0, 7))

        # ── 발화 텍스트 (항상 수동) ────────────────────────────────
        speech = cmd.get('speech', '')
        if 'Ÿ' in speech:
            speech = ''
        new_speech = self._prompt_str('  발화 텍스트', speech)
        corrected_speech = new_speech if new_speech else speech

        # ── 관절 교정 ─────────────────────────────────────────────
        if method == 'D':
            self._kinesthetic_capture(corrected_action)
        else:
            print(f'\n  {LINE[:30]}')
            print('  관절 값 입력 (Enter = 현재값 유지)')
            for field, lo, hi in [
                ('head_tilt',       1500, 3086),
                ('head_pan',        1043, 3071),
                ('head_roll',       1630, 2452),
                ('r_arm_pitch',     1024, 2451),
                ('l_arm_pitch',     37,   1542),
                ('r_shoulder_roll', 1000, 2050),
                ('r_elbow_pitch',   2047, 3062),
                ('l_shoulder_roll', 1047, 2056),
                ('l_elbow_pitch',   1021, 2007),
            ]:
                cur = action.get(field, (lo + hi) // 2)
                val = self._prompt(f'  {field} [{lo}-{hi}] 현재={cur}', cur)
                corrected_action[field] = int(np.clip(int(val), lo, hi))

        # ── 바퀴 (항상 수동: 속도 제어, 물리 시연 무의미) ──────────
        print('\n  바퀴 속도 (이동 불필요시 0 입력)')
        for field in ('right_wheel', 'left_wheel'):
            cur = action.get(field, 0.0)
            val = self._prompt(f'  {field} [-300~300] 현재={cur:.0f}', cur)
            corrected_action[field] = float(np.clip(float(val), -300, 300))

        print(LINE)

        corrected_cmd = dict(cmd)
        corrected_cmd['expression_id'] = corrected_expr_id
        corrected_cmd['action']        = corrected_action
        corrected_cmd['speech']        = corrected_speech

        self._publish_command(corrected_cmd)
        self._log_step(cmd, corrected_cmd, is_corrected=True)
        self._pending = None
        self._state = 'IDLE'
        print(f'[OK] 교정 완료 — 스텝 {self._writer.current_step - 1} 저장됨\n')
        sys.stdout.flush()

    def _kinesthetic_capture(self, corrected_action: dict):
        """토크 OFF → 사용자가 로봇을 직접 조작 → Enter → 인코더 값 캡처."""
        torque_msg = Bool()

        torque_msg.data = False
        self.pub_torque.publish(torque_msg)

        print(f'\n{LINE}')
        print(' [키네스테틱] 토크 OFF — 원하는 자세로 로봇을 직접 조작하세요')
        print(' 완료되면 Enter를 누르세요')
        print(LINE)
        sys.stdout.flush()
        input()  # 사용자가 자세 조작 후 Enter

        torque_msg.data = True
        self.pub_torque.publish(torque_msg)

        # action_node가 인코더 읽고 /haru_joints/state 발행할 때까지 대기
        # (MultiThreadedExecutor 덕분에 이 블로킹 중에도 _joints_cb 실행됨)
        capture_start = time.time()
        deadline = capture_start + 2.0
        joints = None
        while time.time() < deadline:
            with self._joints_lock:
                if self._latest_joints_time > capture_start:
                    joints = dict(self._latest_joints)
                    break
            time.sleep(0.05)

        if joints is None:
            print('  [경고] 관절 위치 수신 실패 (action_node가 실행 중인지 확인).')
            print('         현재 제안값을 유지합니다.')
            sys.stdout.flush()
            return

        for name in POSITION_JOINT_ORDER:
            corrected_action[name] = joints[name]

        print('  캡처 완료:')
        print(f'    머리  : tilt={joints["head_tilt"]:.0f}'
              f'  pan={joints["head_pan"]:.0f}'
              f'  roll={joints["head_roll"]:.0f}')
        print(f'    오른팔: pitch={joints["r_arm_pitch"]:.0f}'
              f'  shoulder={joints["r_shoulder_roll"]:.0f}'
              f'  elbow={joints["r_elbow_pitch"]:.0f}')
        print(f'    왼팔  : pitch={joints["l_arm_pitch"]:.0f}'
              f'  shoulder={joints["l_shoulder_roll"]:.0f}'
              f'  elbow={joints["l_elbow_pitch"]:.0f}')
        sys.stdout.flush()

    # ── 입력 헬퍼 ────────────────────────────────────────────────

    def _prompt(self, label: str, default) -> float:
        try:
            raw = input(f'{label}: ').strip()
            return float(raw) if raw else float(default)
        except (ValueError, EOFError):
            return float(default)

    def _prompt_str(self, label: str, default: str) -> str:
        try:
            raw = input(f'{label} (현재={default!r}): ').strip()
            return raw if raw else default
        except EOFError:
            return default

    # ── 에피소드 종료 ────────────────────────────────────────────

    def _end_episode(self, accepted: bool):
        saved = self._writer.end_episode(accepted=accepted)
        self._in_episode = False
        self._state = 'IDLE'
        self._pending = None
        if saved:
            print(f'\n[SAVED] 에피소드 {self._writer.episode_id} 저장 완료  '
                  f'({self._writer.step_count}스텝)\n')
        else:
            print('\n[CANCEL] 에피소드 취소됨 (데이터 삭제)\n')
        print(' [N] 새 에피소드 시작   [Q] 종료')
        sys.stdout.flush()

    # ── 발행 / 저장 헬퍼 ─────────────────────────────────────────

    def _publish_command(self, cmd: dict):
        msg = String()
        msg.data = json.dumps(cmd, ensure_ascii=False)
        self.pub_cmd.publish(msg)

        expr_msg = Int32()
        expr_msg.data = int(cmd.get('expression_id', 0))
        self.pub_expr.publish(expr_msg)

    def _log_step(self, vla_cmd: dict, final_cmd: dict, is_corrected: bool):
        image = self._latest_image if self._latest_image is not None \
                else np.zeros((448, 896, 3), dtype=np.uint8)

        def _merge_expr(cmd: dict) -> dict:
            action = dict(cmd.get('action', {}))
            action['expression_id'] = cmd.get('expression_id', 0)
            return action

        self._writer.save_step(
            image=image,
            vla_action=_merge_expr(vla_cmd),
            final_action=_merge_expr(final_cmd),
            language_instruction=final_cmd.get('speech', ''),
            speech_text=final_cmd.get('speech', ''),
            emotion=final_cmd.get('emotion', 'neutral'),
            is_corrected=is_corrected,
            attention_source=vla_cmd.get('attention_source', ''),
            attention_context=vla_cmd.get('attention_context', ''),
        )


def main(args=None):
    rclpy.init(args=args)
    node = HaruHITLNode()
    # MultiThreadedExecutor: _run_correction이 input()으로 블록 중에도
    # _joints_cb (ReentrantCallbackGroup)가 별도 스레드에서 실행됨
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
