# FeTS 2022 Federated Learning Execution Walkthrough

This document guides you through running the Centralized Baseline and the Federated Learning (Flower AI) experiments for the FeTS 2022 dataset. It also details the experimental setup, model architecture, and metrics collection.

---

## 1. Experimental Setup & Design

### 1.1 Model Architecture: Swin UNETR
We use the **Swin UNETR** (Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images) architecture. 
- **Why Swin UNETR?** It is a state-of-the-art 3D Vision Transformer tailored for medical image segmentation. Unlike standard CNNs (e.g., 3D U-Net) which rely heavily on localized convolutional kernels, Swin UNETR uses a hierarchical Swin Transformer as the encoder to extract multi-scale representations, effectively modeling long-range contextual information.
- **Input:** The model natively handles 3D data. It accepts 3D patches of size `128x128x128` with 4 channels corresponding to the 4 input MRI modalities (FLAIR, T1, T1c, T2).
- **Output:** The model outputs a 3-channel 3D segmentation map representing the probabilities for three sub-regions: Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET).

### 1.2 Dataset Split & Partitioning
The data used is the **FeTS 2022 Challenge Dataset**. Regardless of the training approach, the underlying patient data is subjected to a strict **80-10-10 patient-level split** for Training, Validation, and Testing. 

**Achieving Non-IID Distribution:**
To emulate a true Federated Learning environment, we use the `partitioning_1.csv` file to distribute the dataset among 3 distinct clients. This scheme creates a highly imbalanced, non-IID (Independent and Identically Distributed) data distribution that mimics distinct hospital sizes and demographics:
- **Client 1:** Receives all data from Partition 1 (a large institution with ~511 subjects).
- **Client 2:** Receives all data from Partition 18 (a moderately large institution with ~382 subjects).
- **Client 3:** Receives an aggregation of all remaining partitions (a collection of smaller clinics with < 50 subjects each).

### 1.3 Centralized vs. Federated Training
- **Centralized Training Baseline:** Simulates an environment where all data from all clients is pooled into a single, global dataset. It acts as the "upper-bound" performance benchmark.
- **Federated Training:** Employs **Flower AI** to orchestrate training across 3 decentralized nodes. Each client trains exclusively on its local, non-IID partition for 1 local epoch per round. Afterward, clients send their model parameters back to the server, which aggregates them using the `FedAvg` strategy. This process is repeated for 100 rounds.

### 1.4 Fair Comparison Guarantee
To guarantee that the comparison between the Federated approach and the Centralized baseline is scientifically sound and fair, the centralized model and the FL model differ **only** in the training paradigm (one machine sees all data vs distributed data).

**Strict Split Preservation:**
Every patient is assigned to exactly one set: Train, Val, or Test. The Centralized model does *not* create a new random 80/10/10 split over the global dataset. Instead, it reuses the exact splits computed independently for each FL client. This mathematically guarantees:
- `Centralized_Train` = `Client1_Train` + `Client2_Train` + `Client3_Train`
- `Centralized_Val` = `Client1_Val` + `Client2_Val` + `Client3_Val`
- `Centralized_Test` = `Client1_Test` + `Client2_Test` + `Client3_Test`

**Other Guarantees:**
- Both approaches share the exact same 3D preprocessing pipeline (e.g., orientation transforms, spacing normalization, cropping, and intensity normalization).
- They use identical model architecture configurations (Swin UNETR feature sizes and channels), AdamW optimizers, and learning rate schedules.
- They optimize the exact same objective: `DiceCELoss` (a combination of Dice Loss and Cross-Entropy Loss).

---

## 2. Metrics & Evaluation

Metrics are heavily prioritized in this pipeline and are strictly logged to CSV files throughout both training runs.

### 2.1 Segmentation Performance Metrics
At the end of every epoch (in centralized) or round (in federated), the models are evaluated on the validation sets. Because FeTS provides raw labels (1=NCR/NET, 2=ED, 4=ET), the labels and predictions are dynamically converted into 3 distinct binary channels to accurately calculate region-specific Dice scores:
- **Dice WT (Whole Tumor):** Evaluates all tumor classes combined (Classes 1, 2, and 4).
- **Dice TC (Tumor Core):** Evaluates the tumor core (Classes 1 and 4).
- **Dice ET (Enhancing Tumor):** Evaluates the enhancing tumor region exclusively (Class 4).
- **Mean Dice:** The arithmetic mean of the WT, TC, and ET Dice scores.

These calculations are driven by custom PyTorch tensor evaluations inside the `utils/metrics_utils.py` module.

### 2.2 System & Resource Metrics
To quantify system efficiency and the overhead introduced by FL, a background `ResourceMonitor` thread runs asynchronously during training. It polls the operating system every second to track:
- **Peak RAM (MB):** Tracked via Python's `psutil` library.
- **Peak VRAM (MB) & GPU Utilization (%):** Tracked using `nvidia-smi` subprocesses.

---

## 3. Execution Instructions

### Setup Requirements
```bash
pip install -r requirements.txt
```

> [!IMPORTANT]
> The scripts assume the FeTS dataset is located in `data/FeTS2022` and that the `MICCAI_FeTS2022_TrainingData` folder and `partitioning_1.csv` file exist inside.

### 3.1 Centralized Baseline Training
To train the centralized model on the entirety of the dataset:
```bash
python -m train.centralized_baseline
```
- **Output Model:** The best model checkpoint is saved to `checkpoint/swin_unetr_centralized_best.pth`.
- **Metrics:** Training and resource metrics are logged to `train/centralized_baseline_metrics.csv`.

### 3.2 Federated Learning (Flower AI)
To run the FL experiment, start the Server first, followed by the three clients.

**Step 3.2.1: Start the Flower Server**
In your first terminal or SLURM node, run:
```bash
python -m fl.server --rounds 100 --server_address 0.0.0.0:8080
```
- **Output:** The server aggregates weights and saves a global checkpoint to `checkpoint/swin_unetr_fl_round_X.pth` at the end of each round.

**Step 3.2.2: Start the Clients**
Launch three clients simultaneously on separate nodes. Replace `<SERVER_IP>` with the server node's IP address (use `127.0.0.1` if running locally).

**Client 1:**
```bash
python -m fl.client --client_id 1 --server_address <SERVER_IP>:8080
```
**Client 2:**
```bash
python -m fl.client --client_id 2 --server_address <SERVER_IP>:8080
```
**Client 3:**
```bash
python -m fl.client --client_id 3 --server_address <SERVER_IP>:8080
```
- **Metrics:** Each client locally tracks its own training loss and system resource utilization (GPU, VRAM, RAM) and logs them into `fl/client_<id>_metrics.csv`.

> [!TIP]
> The server will not start the first training round until all 3 clients have successfully connected.
