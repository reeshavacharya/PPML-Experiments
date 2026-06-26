# System Model and Adversary Model

## System Model

**Setting.** We consider a cross-silo federated learning system for privacy-preserving brain tumor segmentation across geographically distributed medical institutions. The system is designed to enable collaborative model training on sensitive neuroimaging data without requiring any institution to share raw patient scans — a prerequisite for multi-site clinical decision support under regulations such as HIPAA and GDPR.

### Participants

The system comprises a central *aggregation server* $\mathcal{S}$ and a fixed set of $K = 3$ institutional *clients* $\mathcal{C} = \{C_1, C_2, C_3\}$. Each client $C_k$ represents an autonomous clinical site that holds a private, non-overlapping partition of multi-modal brain MRI studies from the FeTS 2022 dataset (Pati et al., 2022). The data partitioning reflects real-world institutional provenance:

| Client | Partition Logic | Test Samples | Institutional Character |
|--------|----------------|--------------|------------------------|
| $C_1$ | Partition ID = 1 (single site) | 52 | Large single-institution cohort |
| $C_2$ | Partition ID = 18 (single site) | 39 | Moderate single-institution cohort |
| $C_3$ | All remaining partition IDs | 36 | Aggregation of smaller contributing sites |

This partitioning induces **natural non-IID heterogeneity**: each institution's data reflects its own scanner hardware, acquisition protocols, patient demographics, and annotation conventions — a realistic distribution shift that is central to the experimental design.

### Data and Task

Each patient study consists of four co-registered MRI modalities (FLAIR, T1, T1ce, T2) as 3D NIfTI volumes (native resolution $240 \times 240 \times 155$ voxels). The segmentation task targets three hierarchical BraTS tumor sub-regions: Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET). Each client independently maintains an 80/10/10 train/validation/test split (deterministic seed), ensuring evaluation integrity across sites.

### Model Architecture

All clients share an identical Swin UNETR architecture (Hatamizadeh et al., 2022) instantiated via MONAI, with 4 input channels, 3 output channels, and feature size 48 (~62M parameters). Training operates on randomly cropped $128 \times 128 \times 128$ patches with standard augmentations (random flips, 90-degree rotations, intensity shifts). The loss function is a composite Dice-Cross-Entropy loss; optimization uses AdamW ($\text{lr} = 3 \times 10^{-4}$, weight decay $= 10^{-5}$).

### Federated Protocol

Training follows the FedAvg algorithm (McMahan et al., 2017) implemented via the Flower framework over $T = 100$ communication rounds:

1. **Broadcast:** At round $t$, the server transmits the current global model parameters $\theta^t$ to all $K$ clients.
2. **Local Training:** Each client $C_k$ initializes from $\theta^t$ and performs $E = 1$ local epoch of SGD over its private training set, producing updated parameters $\theta_k^{t+1}$.
3. **Upload:** Each client transmits $\theta_k^{t+1}$ (full model update) to the server.
4. **Aggregation:** The server computes a weighted average:

$$\theta^{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} \theta_k^{t+1}$$

where $n_k$ is the number of training examples at client $C_k$ and $n = \sum_k n_k$.

5. **Evaluation:** After aggregation, the server distributes $\theta^{t+1}$ to all clients for validation; the model achieving the highest weighted-average Dice score is checkpointed.

### Communication and Infrastructure

Client-server communication uses gRPC (Flower's default transport) over a private institutional cluster network. Each client runs on a dedicated compute node (16 CPU cores, 64 GB RAM, 1 GPU), while the server runs on a CPU-only node (8 cores, 32 GB RAM). All nodes are co-located within a single SLURM-managed HPC cluster, connected via a private high-bandwidth interconnect. The server requires all $K = 3$ clients to be available in every round (synchronous protocol; no stragglers tolerated).

### Trust Assumptions

The server is assumed to be *honest-but-curious* — it faithfully executes the aggregation protocol but may attempt to infer information from the model updates it receives. Clients are assumed to be *honest* — they follow the prescribed training procedure and do not tamper with their local updates. Raw patient data never leaves the institutional boundary of any client; only model parameters (numerical weight tensors) are communicated. No additional privacy-enhancing technologies (differential privacy, secure aggregation, homomorphic encryption) are applied in this baseline configuration.

---

## Adversary Model

We define the adversary model with respect to the threat landscape of cross-silo federated learning in healthcare, specifying the adversary's **identity**, **capabilities**, **goals**, and **knowledge**.

### Adversary Identity

We consider two classes of adversaries:

- **A1 — Compromised Aggregation Server.** An honest-but-curious server that faithfully performs FedAvg aggregation but attempts to extract private patient information from the model updates it observes. This is the primary threat model, as the server sees all $K$ clients' parameter updates in plaintext at every round $t \in [1, T]$.

- **A2 — Compromised Client (Insider Threat).** An honest-but-curious client $C_j$ that participates correctly in the protocol but attempts to infer information about the private data of other clients $C_{k \neq j}$ from the global model parameters it receives.

### Adversary Capabilities

| Capability | A1 (Server) | A2 (Client) |
|-----------|-------------|-------------|
| Observe global model $\theta^t$ at each round | Yes | Yes |
| Observe individual client updates $\theta_k^{t+1}$ | Yes, all $k$ | No (only $\theta^{t+1}$ after aggregation) |
| Compute gradient residuals $\Delta_k^t = \theta_k^{t+1} - \theta^t$ | Yes | No |
| Access auxiliary data from same domain | Possible | Yes (own training data) |
| Modify the aggregation protocol | No (honest-but-curious) | N/A |
| Inject malicious updates | N/A | No (honest) |
| Access the communication channel (eavesdropping) | Inherent | No (private cluster network) |

### Adversary Goals

The adversary's objectives are information-theoretic rather than disruptive:

1. **Membership Inference.** Determine whether a specific patient's MRI study was part of a particular client's training set — a direct HIPAA/GDPR concern in multi-site clinical collaborations.

2. **Property Inference.** Infer sensitive aggregate properties of a client's data distribution, such as institutional demographics, tumor subtype prevalence, or scanner characteristics, from the structure of model updates.

3. **Data Reconstruction.** Recover (approximate) raw MRI volumes from observed gradient updates, e.g., via gradient inversion attacks (Zhu et al., 2019). For 3D medical volumes with 4 modalities at $128^3$ resolution (~8.4M voxels per sample), this represents a high-dimensional reconstruction problem.

### Adversary Knowledge

The adversary is assumed to know:

- The model architecture (Swin UNETR), hyperparameters, and loss function (public information).
- The federated protocol (FedAvg with $E = 1$, $T = 100$).
- The dataset provenance (FeTS 2022) and partitioning scheme.
- The number of clients and their approximate dataset sizes.

The adversary does **not** know:

- The raw imaging data at any client (the asset being protected).
- The exact sample-level composition of any client's train/val/test split.
- Per-sample gradients (only the aggregated update after a full local epoch is visible to A1).

### Security Analysis of the Current Configuration

The current system provides **privacy-by-architecture** (data never leaves institutional boundaries) but does not employ formal privacy guarantees. Specifically:

| Privacy Mechanism | Status | Implication |
|-------------------|--------|-------------|
| Data locality | **Active** | Raw MRI data never transmitted; only ~62M float32 parameters exchanged per round |
| Differential Privacy (DP) | Not applied | Model updates may leak information about individual training samples |
| Secure Aggregation | Not applied | Server A1 observes individual client updates in plaintext |
| Communication Encryption (TLS) | Not enforced at application layer | gRPC over private cluster network; relies on network-level isolation |
| Gradient Compression/Sparsification | Not applied | Full model parameters transmitted, maximizing information available to adversary |

**Practical risk assessment.** The single local epoch ($E = 1$) with batch size 1 means each client's update is a full pass over its private training set before transmission — this aggregates gradients over many samples, providing some empirical privacy amplification compared to per-batch gradient sharing. The high dimensionality of 3D medical volumes (>8M voxels) and the use of a large transformer model (~62M parameters) further increase the difficulty of exact data reconstruction attacks. However, without formal DP guarantees ($\epsilon$-differential privacy), no provable bound on information leakage can be stated.

This baseline configuration establishes the performance-privacy frontier against which future enhancements (e.g., local DP with Gaussian noise, secure aggregation via secret sharing, or gradient compression) can be quantitatively evaluated in terms of their segmentation accuracy trade-off (Dice score degradation) versus formal privacy guarantees.
