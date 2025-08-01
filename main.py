"""
EEG Source Localization with Variational Autoencoder
Main execution script
"""

import numpy as np
import torch
from models import EEGEncoder, EEGDecoder, reparameterize
from data_generator import generate_eeg_dataset, analyze_predictions


def main():
    """Main execution function"""
    
    # 1. 데이터 생성
    print("Generating EEG dataset...")
    all_eeg, all_labels = generate_eeg_dataset(n_trials=1000)
    
    # 2. 모델 설정
    n_eeg_channels = len(all_eeg[0])  # EEG 채널 수
    n_vertices = 8196  # 예시 vertex 수 (실제로는 src에서 가져와야 함)
    latent_dim = 32
    
    encoder = EEGEncoder(input_dim=n_eeg_channels, latent_dim=latent_dim)
    decoder = EEGDecoder(latent_dim=latent_dim, output_dim=n_vertices)
    
    # 3. 예시 추론 (실제로는 학습 후 사용)
    print("Running example inference...")
    
    # 첫 번째 EEG 샘플로 테스트
    eeg_tensor = torch.tensor(all_eeg[0], dtype=torch.float32)
    
    # 인코더로 잠재 공간 추출
    mu, logvar = encoder(eeg_tensor)
    
    # 재매개화 트릭으로 샘플링
    z_samples = reparameterize(mu, logvar, n_samples=100)
    
    # 디코더로 source 예측
    preds = decoder(z_samples)
    preds_np = preds.detach().cpu().numpy()
    
    # 4. 결과 분석
    print("Analyzing predictions...")
    results = analyze_predictions(preds_np)
    
    print(f"Prediction shape: {preds_np.shape}")
    print(f"Mean activation range: [{results['mean'].min():.3f}, {results['mean'].max():.3f}]")
    print(f"Std activation range: [{results['std'].min():.3f}, {results['std'].max():.3f}]")
    
    print("Done!")


if __name__ == "__main__":
    main()
