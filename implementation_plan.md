# Federated Learning Experiment Design for FeTS 2022

This plan describes the complete overhaul of the current implementation to support the Federated Learning (FL) experiment on the FeTS 2022 dataset using Flower AI, as described in the provided document.

## User Review Required
> [!IMPORTANT]
> The document specifies using a standard **ViT-Base** for **Segmentation**. Standard ViT-Base architectures are typically for image classification. To perform segmentation, we either need a segmentation head on top of the ViT backbone (e.g., SETR, UNETR) or a different architecture. I will implement a ViT-based architecture adapted for segmentation with a basic decoding head unless you specify a particular architecture (e.g., UNETR).
>
> Additionally, standard ViT accepts 3-channel (RGB) inputs, but FeTS 2022 provides 4 MRI modalities. I will modify the patch embedding layer to accept 4-channel input. 

> [!IMPORTANT]
> Since this will run on a SLURM cluster across multiple nodes, the server address must be dynamically resolved or statically assigned, and the SLURM submission scripts will need to handle the orchestration (launching the server first, getting its IP, and then launching the clients with that IP). I will provide the necessary SLURM `.sh` scripts. Please verify if your SLURM environment has specific constraints (e.g., partition names, account names).

## Open Questions
- **SLURM Setup:** Do you have specific SLURM partition names, queue names, or module loads required for PyTorch/CUDA in your cluster?
- **Data Dimension:** The FeTS dataset provides 3D MRI volumes. The document mentions "resized MRI slices" (2D). Should we extract 2D slices along a specific axis (e.g., axial) to train a 2D ViT, or should we use a 3D ViT model? Extracting 2D slices is standard for 2D ViTs.
- **Centralized Baseline:** Do you want the centralized baseline script (`server_centralized.py` or similar) to be created in this initial implementation, or should we focus strictly on the FL components first?

## Proposed Changes

We will restructure the project to separate FL clients, the FL server, data loading logic for FeTS, and SLURM scripts.

---

### Data Loading & Processing

#### [NEW] [fets_dataset.py](file:///Users/reeshavacharya/CIRCE/PPML-Experiments-FL/data_loader/fets_dataset.py)
- **Purpose**: PyTorch Dataset for FeTS 2022.
- **Responsibilities**:
  - Load 3D NIfTI files (`flair.nii.gz`, `t1.nii.gz`, `t1ce.nii.gz`, `t2.nii.gz`) and the segmentation mask.
  - Slicing: Convert 3D volumes into 224x224 2D slices to match the ViT input resolution.
  - Parse `partitioning_1.csv` to filter patients belonging to specific partitions.
  - Implement 80-10-10 patient-level train/val/test split.

---

### Model Architecture

#### [MODIFY] [vit_base.py](file:///Users/reeshavacharya/CIRCE/PPML-Experiments-FL/models/vit_base.py)
- **Purpose**: Update the existing ViT model for segmentation and 4-channel input.
- **Responsibilities**:
  - Modify the patch embedding convolution to take `in_channels=4`.
  - Add a segmentation decoder head to output predictions for the 3 sub-regions (WT, TC, ET).

---

### Federated Learning Infrastructure

#### [NEW] [client.py](file:///Users/reeshavacharya/CIRCE/PPML-Experiments-FL/fl/client.py)
- **Purpose**: Flower AI client script.
- **Responsibilities**:
  - Receive client ID (1, 2, or 3) via command line.
  - Load the appropriate dataset subset based on the client mapping (Client 1: Partition 1, Client 2: Partition 18, Client 3: Remaining).
  - Implement `flwr.client.NumPyClient` with `get_parameters`, `fit`, and `evaluate` methods.
  - Run local training (1 epoch per round) using Dice Loss / CE + Dice.
  - Track local system metrics (GPU, CPU, RAM) and local training dynamics (Loss, Dice).

#### [NEW] [server.py](file:///Users/reeshavacharya/CIRCE/PPML-Experiments-FL/fl/server.py)
- **Purpose**: Flower AI server script.
- **Responsibilities**:
  - Initialize the `FedAvg` strategy.
  - Coordinate 3 clients for 100 rounds.
  - Track communication metrics (round time, payload sizes).
  - Aggregate evaluation metrics across clients.
  - Save global model checkpoints.

---

### Metrics & Evaluation

#### [NEW] [metrics_utils.py](file:///Users/reeshavacharya/CIRCE/PPML-Experiments-FL/utils/metrics_utils.py)
- **Purpose**: Comprehensive metric tracking.
- **Responsibilities**:
  - Compute Segmentation metrics: Dice (WT, TC, ET), Mean Dice, IoU, Hausdorff95, Sensitivity, Specificity, Precision.
  - Track System/Resource metrics using an enhanced version of the `ResourceMonitor` from the baseline.

---

### Orchestration

#### [NEW] [run_fl_slurm.sh](file:///Users/reeshavacharya/CIRCE/PPML-Experiments-FL/slurm/run_fl_slurm.sh)
- **Purpose**: SLURM batch script to launch the server and 3 clients on different nodes.
- **Responsibilities**:
  - Allocate 4 nodes (1 server, 3 clients).
  - Discover the IP address of the server node.
  - Start `server.py` on the server node.
  - Start `client.py --client_id 1/2/3 --server_address <IP>` on the client nodes.

## Verification Plan

### Automated Tests
- Run a dry-run local test with mock data or a very small subset (1 batch) to ensure the ViT forward pass, loss computation, and Flower client-server communication work without SLURM.
- `python -m fl.server &` followed by `python -m fl.client --client_id 1` etc.

### Manual Verification
- Deploy the SLURM script `run_fl_slurm.sh` to the cluster and verify that:
  - 4 jobs/tasks are started across different nodes.
  - The clients successfully connect to the server.
  - GPU utilization and RAM are logged properly.
  - Checkpoints and CSV metric logs are generated per round.
