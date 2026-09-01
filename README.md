# NeuralStream: Real-Time BCI Signal Decoding

### Project Overview
NeuralStream is a high-performance deep learning pipeline designed to decode motor imagery from raw, non-stationary EEG signals. This project implements a modified **EEGNet** architecture to extract neural intent (Left Hand, Right Hand, Feet, Tongue) from high-noise environments, bridging the gap between computational neuroscience and real-time AI engineering.

### Technical Implementation
Implemented **Early Stopping** and **Label Smoothing** to prevent overfitting. Achieved a significant generalization improvement, moving from 31% to 40%+ on unseen session data, proving the model learned robust physiological features rather than session-specific noise.

### Key Engineering Highlights
*   **Multi-Subject Training:** Trained on a diverse pool of subjects (Subjects 1, 2, and 3) to force the model to learn invariant brain patterns rather than individual session quirks.
*   **Signal Normalization:** Implemented a custom `StandardScaler` pipeline to handle biological variance in signal amplitude across different recording sessions.
*   **Defensive Regularization:** Applied high-rate Dropout (0.75) and Gaussian Noise Augmentation to break memorization and encourage generalization.

### Results
*   **Training Accuracy:** 68.75% (Stabilized to prevent curve-fitting).
*   **Real-World Generalization:** 
    *   **Subject 1 (Unseen):** 41.67%
    *   **Subject 2 (Unseen):** 43.75%
    *   **Subject 3 (Unseen):** 39.58%
    *   *All results significantly outperform the 25% random baseline.*

### Tech Stack
*   **Deep Learning:** PyTorch
*   **Neuroscience:** MNE-Python, MOABB
*   **Data Science:** Scikit-learn, NumPy, Matplotlib

### Quant Connection
The signal processing and regularization techniques used here—specifically **StandardScaler normalization**, **Gaussian Noise stress-testing**, and **handling non-stationary time-series data**—are the exact same principles applied to my **Live Volatility Trading System** to ensure model stability in shifting market regimes.
