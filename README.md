# IF3270 Tubes 2 - Dicarry-Udil

Implementation of CNN, Simple RNN, and LSTM from scratch using NumPy for image classification and image captioning.

## Project Structure
Refer to `DEV_WORKFLOW.md` for detailed work distribution and the critical path.

## Setup Instructions

### 1. Environment
```bash
# Clone the repository
git clone <repository-url>
cd Tubes2_Dicarry-Udil

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

#### Track A: Intel Image Classification
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).
2. Place the downloaded `archive.zip` at `data/intel/archive.zip`.
3. Run the setup script to extract and split the data (creates 80/20 train/val split):
   ```bash
   python3 setup_intel.py
   ```

#### Track B: Flickr8k
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k).
2. Place the downloaded `archive.zip` at `data/flickr8k/archive.zip`.
3. Run the setup script to extract and organize the data:
   ```bash
   python3 setup_flickr8k.py
   ```

## Development Workflow
Detailed distribution of tasks among team members (Member A, B, and C) can be found in [DEV_WORKFLOW.md](./DEV_WORKFLOW.md).
