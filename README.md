# FareCaster

## Flight Price Forecasting with Deep Learning

FareCaster is a machine learning system designed to predict future flight prices using various recurrent neural network architectures. The project implements and compares multiple time series forecasting approaches including GRU, LSTM, and Transformer-based models.

## Features

- Multiple neural network architectures for time series forecasting
- Sliding window approach for sequence data preparation
- Comprehensive evaluation metrics and model comparison
- Experiment tracking and visualization
- Hyperparameter optimization support

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/FareCaster.git
cd FareCaster

# Create and activate a virtual environment (optional but recommended)
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
FareCaster/
├── models.py              # Neural network model architectures
├── train.py               # Training pipeline and experiment management
├── preprocess.py          # Data preprocessing utilities
├── constants.py           # Project-wide constants and configurations
├── utils/
│   └── logging.py         # Logging and visualization utilities
├── data/                  # Data directory (not tracked in git)
│   ├── raw/               # Raw datasets
│   └── processed/         # Processed datasets
├── saved/                 # Saved model checkpoints and experiment results
├── notebooks/             # Jupyter notebooks for exploration and analysis
│   ├── EDA.ipynb          # Exploratory data analysis
│   └── data_processing.ipynb  # Data processing workflows
└── README.md              # Project documentation
```

## Usage

### Data Preparation

1. Place your flight price data in `data/`
2. Define your global variables and filenames in constants.py and run the preprocessing script:
   ```bash
   python preprocess.py
   ```

### Training Models

```bash
python train.py
```

You can modify hyperparameters in the main section of train.py.

## Model Architectures

FareCaster implements several neural network architectures:

1. **GRU Model**: Simple GRU-based recurrent network
2. **LSTM Model**: Basic LSTM network for sequence modeling
3. **Deep GRU/LSTM Models**: Multi-layer versions with dropout regularization
4. **LSTM Encoder-Decoder**: Sequence-to-sequence architecture that separates encoding of historical data from prediction generation
5. **Transformer Model**: Attention-based architecture for capturing long-range dependencies

Several State of the Art (SOTA) models are also experimented with in preprocess_sota.py and train_sota.py.

## Evaluation

Models are evaluated using:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

Results are saved to the saved directory with timestamps for easy comparison.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- PyTorch team for the deep learning framework
- [Your data sources and inspirations]

Similar code found with 2 license types