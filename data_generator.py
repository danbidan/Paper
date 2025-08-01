import numpy as np
import mne
from pathlib import Path
from eeg_generator import EEGGenerator


def setup_eeg_generator():
    """Setup EEG generator with MNE sample data"""
    subjects_dir = mne.datasets.sample.data_path() / 'subjects'
    
    model = mne.make_bem_model(subject='sample', conductivity=(0.3, 0.006, 0.3), ico=4, subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    
    gen = EEGGenerator(
        subjects_dir=subjects_dir,
        montage='standard_1020',
        sfreq=200,
        spacing='ico4',
        conductivity=(0.3, 0.006, 0.3),
        bem=bem
    )
    
    # Pick dipole candidates
    gen.pick_candidate_from_surface_src()
    
    return gen


def generate_eeg_dataset(n_trials=1000):
    """Generate EEG dataset with random dipole sources"""
    gen = setup_eeg_generator()
    
    all_eeg = []
    all_labels = []
    
    for i in range(n_trials):
        # 1) dipole 위치 & label 생성
        dipoles = gen.dipoles_in_src(n_sources=1, extents=(10.5, 29))
        labels = gen.create_labels(n_sources=1, extents=(10.5, 29), dipoles_vertex_coor_dis_paired_src=dipoles)

        # 2) 시계열 파형 생성: 랜덤 주파수 & 진폭
        sfreq = gen.sfreq
        duration = 1.0
        n_samples = int(sfreq * duration)
        times = np.linspace(0, duration, n_samples)

        freq = np.random.uniform(8, 12)                # 예: 8~12Hz 랜덤
        amp = np.random.uniform(2e-9, 5e-9)             # 진폭도 랜덤
        start = np.random.uniform(0.3, 0.6)
        end = start + 0.1                               # 100ms 짜리
        start_idx = int(start * sfreq)
        end_idx = int(end * sfreq)

        waveform = np.zeros((1, n_samples))
        waveform[0, start_idx:end_idx] = amp * np.sin(2 * np.pi * freq * times[start_idx:end_idx])

        # 3) Raw 생성 및 EEG 시뮬레이션
        raw = gen.create_raw(labels=labels, amplitude_kernel=0, time_course=waveform)
        eeg = gen.generate_eeg(raw, snr_db=20, filter=(1, 40))

        # 4) 저장
        all_eeg.append(eeg[0])       # 채널 0만 저장 예시
        all_labels.append((dipoles[0], freq, amp))  # dipole, 주파수, 진폭 정보 저장
    
    return all_eeg, all_labels


def analyze_predictions(preds_np):
    """Analyze model predictions and compute statistics"""
    mean_map = preds_np.mean(axis=0)         # vertex-wise 평균
    std_map = preds_np.std(axis=0)           # vertex-wise 표준편차
    ci_95_low = np.percentile(preds_np, 2.5, axis=0)
    ci_95_high = np.percentile(preds_np, 97.5, axis=0)
    
    return {
        'mean': mean_map,
        'std': std_map,
        'ci_95_low': ci_95_low,
        'ci_95_high': ci_95_high
    } 