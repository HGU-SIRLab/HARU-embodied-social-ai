"""
HARU 에피소드 데이터 → RLDS (TFRecord) 변환 스크립트
워크스테이션(GPU 서버)에서 실행 — Jetson에서 실행 불필요

사용법:
  python scripts/convert_to_rlds.py \
    --episodes_dir data/episodes \
    --output_dir   data/rlds_dataset \
    --dataset_name haru_social_vla

요구 패키지 (워크스테이션):
  pip install tensorflow tensorflow-datasets numpy pillow
"""

import argparse
import json
import numpy as np
from pathlib import Path


def load_episodes(episodes_dir: Path):
    episodes = []
    for ep_dir in sorted(episodes_dir.iterdir()):
        if not ep_dir.is_dir():
            continue
        meta_path = ep_dir / 'metadata.json'
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        steps = []
        for step_path in sorted(ep_dir.glob('step_*.npz')):
            data = np.load(step_path, allow_pickle=True)
            steps.append({
                'image':                np.array(data['image'],   dtype=np.uint8),
                'action':               np.array(data['action'],  dtype=np.float32),
                'action_vla':           np.array(data['action_vla'], dtype=np.float32),
                'is_corrected':         bool(data['is_corrected']),
                'language_instruction': data['language_instruction'].item().decode(),
                'emotion':              data['emotion'].item().decode(),
            })

        if steps:
            episodes.append({'metadata': meta, 'steps': steps})
    return episodes


def convert_to_rlds(episodes_dir: str, output_dir: str, dataset_name: str):
    try:
        import tensorflow as tf
    except ImportError:
        print('[ERROR] TensorFlow 미설치. pip install tensorflow 실행 후 재시도.')
        return

    episodes_dir = Path(episodes_dir)
    output_dir   = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_episodes(episodes_dir)
    print(f'[INFO] 에피소드 {len(episodes)}개 로드됨')

    writer = tf.io.TFRecordWriter(str(output_dir / f'{dataset_name}.tfrecord'))
    total_steps = 0

    for ep_idx, episode in enumerate(episodes):
        steps = episode['steps']
        n = len(steps)

        for i, step in enumerate(steps):
            feature = {
                'observation/image': tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[tf.image.encode_jpeg(step['image']).numpy()]
                    )
                ),
                'action': tf.train.Feature(
                    float_list=tf.train.FloatList(value=step['action'].tolist())
                ),
                'action_vla': tf.train.Feature(
                    float_list=tf.train.FloatList(value=step['action_vla'].tolist())
                ),
                'is_first': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[int(i == 0)])
                ),
                'is_last': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[int(i == n - 1)])
                ),
                'is_corrected': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[int(step['is_corrected'])])
                ),
                'language_instruction': tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[step['language_instruction'].encode()]
                    )
                ),
                'episode_id': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[ep_idx])
                ),
            }
            example = tf.train.Example(
                features=tf.train.Features(feature=feature)
            )
            writer.write(example.SerializeToString())
            total_steps += 1

        print(f'  에피소드 {ep_idx + 1}/{len(episodes)} 변환 완료 ({n} 스텝)')

    writer.close()

    # 데이터셋 요약 저장
    summary = {
        'dataset_name':  dataset_name,
        'total_episodes': len(episodes),
        'total_steps':   total_steps,
        'action_dim':    7,
        'action_space':  [{'name': n, 'min': lo, 'max': hi}
                          for n, lo, hi in [
                              ("expression_id", 0,    7),
                              ("head_tilt",     1500, 3086),
                              ("head_pan",      1043, 3071),
                              ("head_roll",     1630, 2452),
                              ("r_arm_pitch",   1024, 2451),
                              ("right_wheel",  -300,   300),
                              ("left_wheel",   -300,   300),
                          ]],
        'image_size':    [448, 896, 3],
        'output_file':   str(output_dir / f'{dataset_name}.tfrecord'),
    }
    with open(output_dir / 'dataset_info.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f'\n[DONE] 변환 완료')
    print(f'  총 에피소드: {len(episodes)}개')
    print(f'  총 스텝:     {total_steps}개')
    print(f'  저장 위치:   {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes_dir', default='data/episodes')
    parser.add_argument('--output_dir',   default='data/rlds_dataset')
    parser.add_argument('--dataset_name', default='haru_social_vla')
    args = parser.parse_args()

    convert_to_rlds(args.episodes_dir, args.output_dir, args.dataset_name)
