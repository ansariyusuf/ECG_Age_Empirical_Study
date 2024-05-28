# Enhancing ECG-Based Heart Age: Impact of Acquisition Parameters and Generalization Strategies for Varying Signal Morphologies and Corruptions

## Abstract

Electrocardiogram (ECG) is a non-invasive approach to capture the overall electrical activity produced by the contraction and relaxation of the cardiac muscles. It has been established in the literature that the difference between ECG-derived age and chronological age represents a general measure of cardiovascular health. Elevated ECG-derived age strongly correlates with cardiovascular conditions (e.g., atherosclerotic cardiovascular disease). However, the neural networks for ECG age estimation are yet to be thoroughly evaluated from the perspective of ECG acquisition parameters. Additionally, deep learning systems for ECG analysis encounter challenges in generalizing across diverse ECG morphologies in various ethnic groups and are susceptible to errors with signals that exhibit random or systematic distortions.

To address these challenges, we perform a comprehensive empirical study to determine the threshold for the sampling rate and duration of ECG signals while considering their impact on the computational cost of the neural networks. To tackle the concern of ECG waveform variability in different populations, we evaluate the feasibility of utilizing pre-trained and fine-tuned networks to estimate ECG age in different ethnic groups. Additionally, we empirically demonstrate that fine-tuning is an environmentally sustainable way to train neural networks, and it significantly decreases the ECG instances required (by more than 100×) for attaining performance similar to the networks trained from random weight initialization on a complete dataset. Finally, we systematically evaluate augmentation schemes for ECG signals in the context of age estimation and introduce a random cropping scheme that provides best-in-class performance while using shorter-duration ECG signals. The results also show that random cropping enables the networks to perform well with systematic and random ECG signal corruptions.

## Networks Used for Study
- AttiaNet
- ResNet1D

## Key Features and Contributions
- **Empirical Study on ECG Acquisition Parameters:** Detailed analysis of the impact of ECG signal sampling rate and duration on neural network performance and computational cost.
- **Generalization Across Diverse Populations:** Evaluation of pre-trained and fine-tuned networks for estimating ECG age in different ethnic groups.
- **Sustainable Training Methods:** Demonstration of fine-tuning as an environmentally sustainable training method, reducing the need for extensive ECG datasets.
- **Robust Data Augmentation:** Introduction of a random cropping scheme for ECG signals that enhances network performance, especially with corrupted signals.

## Installation
To use the code and reproduce the experiments, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ansariyusuf/ECG-Age-Estimation.git
   cd ECG-Age-Estimation
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Data Preparation
- Download the PTB-XL dataset from the official [PTB-XL repository](https://physionet.org/content/ptb-xl/1.0.3/).
- Preprocess the data as described in the paper.

## Dependencies
- Python 3.x
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you use this code in your research, please cite our paper:
```bibtex
@article{your_paper,
  title={Enhancing ECG-Based Heart Age: Impact of Acquisition Parameters and Generalization Strategies for Varying Signal Morphologies and Corruptions},
  author={Ansari et al.},
  journal={Frontiers in Cardiovascular Medicine},
  year={2024}
}
```

## Acknowledgments
We would like to thank the contributors of the PTB-XL dataset and the developers of the AttiaNet and ResNet1D architectures for their foundational work in ECG age estimation.
