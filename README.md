# GraphDTA-3D: Structure-Aware Drug–Target Binding Affinity Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
## 🧠 Project Summary
This project revisits and extends the GraphDTA model for drug–target binding affinity (DTA) prediction by integrating protein structural information derived from AlphaFold-predicted 3D structures[cite: 1]. The goal is to evaluate whether structure-aware encoding of proteins improves DTA prediction performance across benchmark datasets[cite: 2]. This implementation provides both the baseline GraphDTA model and the enhanced GraphDTA-3D model.

## 🎯 Objectives
* Reproduce the original GraphDTA model using RDKit-based drug graphs and 1D protein sequences[cite: 3].
* Build a GraphDTA-3D model using residue-level protein graphs derived from AlphaFold .pdb files[cite: 4].
* Benchmark both models across two gold-standard datasets: Davis and KIBA[cite: 5].
* Ensure reproducibility via fixed random seeds and consistent dataset splits[cite: 6].
* Evaluate models using test set metrics (RMSE and Concordance Index)[cite: 6].

## ✨ Key Features
* **Baseline GraphDTA:** Faithful reproduction using molecular graphs for drugs and 1D CNNs over protein sequences[cite: 7].
* **GraphDTA-3D:** Enhanced model employing Graph Convolutional Networks (GCNs) for both drug molecular graphs and 3D protein residue-level graphs[cite: 8].
* **Structure-Aware Protein Encoding:** Proteins in GraphDTA-3D are represented as residue graphs, where Cα atoms are nodes and edges are formed based on proximity (<8Å)[cite: 9].
* **Standard Benchmarks:** Evaluated on the Davis and KIBA datasets[cite: 5, 10].
* **Reproducibility Focus:** Scripts and methodology designed for reproducible research[cite: 6].

## 🧪 Methodology

### Model Variants
* **GraphDTA**: Baseline model using molecular graphs (drugs) and 1D CNNs over protein sequences[cite: 7].
* **GraphDTA-3D**: Enhanced model using GCNs over both drug molecular graphs and protein residue-level graphs[cite: 8]. Proteins are represented as residue graphs with Cα atoms as nodes and edges defined by Cα proximity (<8Å)[cite: 9].

### Datasets
The models are benchmarked on two standard DTA datasets:
| Dataset | Size    | Description                                           |
| :------ | :------ | :---------------------------------------------------- |
| Davis   | ~30K    | Kinase-inhibitor pairs with Kd values                 | [cite: 10]
| KIBA    | ~118K   | Aggregated bioactivity scores for kinase inhibitors | [cite: 10]

*(Note: Raw data files like `drugs.csv`, `proteins.csv` (for Davis), and `kiba_affinity_df.csv` (containing SMILES, sequences, and affinity for KIBA) are expected in appropriate subdirectories within `data/raw/{dataset_name}/` as per the preprocessing scripts.)*

### Evaluation Metrics
* **RMSE (Root Mean Squared Error)**: Measures the absolute error in predicted binding affinity[cite: 11]. Lower is better.
* **CI (Concordance Index)**: Measures how well the model ranks pairs[cite: 12]. Higher is better.

### Experimental Setup
* Random seed fixed across Python, PyTorch, and dataloaders for reproducibility[cite: 13].
* Data Split: 80% train / 10% validation / 10% test[cite: 13].
* Optimizer: Adam[cite: 13].
* Batch size: 512[cite: 13].
* Early stopping: Based on validation set Concordance Index (CI)[cite: 13].

## 📈 Results

| Model       | Dataset | Test RMSE ↓ | Test CI ↑ |
| :---------- | :------ | :---------- | :-------- |
| GraphDTA    | Davis   | 0.7382      | 0.7949    | [cite: 14]
| GraphDTA-3D | Davis   | 0.5468      | 0.8810    | [cite: 14]
| GraphDTA    | KIBA    | 0.7427      | 0.7011    | [cite: 14]
| GraphDTA-3D | KIBA    | 0.4187      | 0.8585    | [cite: 14]

GraphDTA-3D consistently outperforms the baseline GraphDTA across both datasets, validating the hypothesis that integrating protein structural information improves DTA prediction[cite: 15].

### Analysis & Discussion
* The 3D structural graphs enable richer encoding of protein interactions by capturing spatial relationships that go beyond mere sequence proximity[cite: 16].
* The performance improvement is more pronounced in the KIBA dataset, which may indicate better generalization capabilities of the structure-aware model on larger and more diverse data[cite: 17].
* The reported test metrics are slightly below some original paper benchmarks[cite: 18]. This is attributed to:
    * No hyperparameter tuning[cite: 18].
    * No ensemble learning[cite: 18].
    * No use of pretrained embeddings (e.g., ProtBERT)[cite: 18].
* However, these results are fully reproducible, honestly reported, and align with common real-world modeling workflows where extensive tuning might not always be feasible initially[cite: 18].

## 📁 Directory Structure

GraphDTA-3D/
├── data/
│   ├── davis/                        # Raw Davis data (e.g., drugs.csv, proteins.csv, drug_protein_affinity.csv)
│   │   └── pdb_files/                # Downloaded PDB structures for Davis proteins
│   ├── kiba/                         # Raw KIBA data (e.g., kiba_affinity_df.csv)
│   │   └── pdb_files/                # Downloaded PDB structures for KIBA proteins
│   ├── external/
│   │   └── uniprot_sprot.fasta       # Local copy of UniProt/Swiss-Prot FASTA (needs to be downloaded)
│   └── processed/
│       ├── davis/
│       │   ├── davis_drug_graphs.pt
│       │   ├── davis_protein_sequence_tensors.pt
│       │   ├── davis_uniprot_mapping.csv
│       │   └── davis_protein_graphs.pt  (For GraphDTA-3D)
│       └── kiba/
│           ├── kiba_drug_graphs.pt
│           ├── kiba_protein_sequence_tensors.pt
│           ├── kiba_uniprot_mapping.csv
│           └── kiba_protein_graphs.pt   (For GraphDTA-3D)
├── notebooks/                        # Original Jupyter notebooks (optional)
├── src/
│   ├── dataloaders/
│   │   ├── init.py
│   │   ├── datasets.py               # GraphDTADataset, GraphDTA3DDataset
│   │   └── custom_collate.py         # collate_fn_graphdta, collate_fn_graphdta3d
│   ├── models/
│   │   ├── init.py
│   │   ├── graphdta_model.py         # GraphDTANet class
│   │   └── graphdta3d_model.py       # GraphDTA3DNet class
│   ├── preprocessing/
│   │   ├── init.py
│   │   ├── process_drug_graphs.py
│   │   ├── process_protein_sequences.py
│   │   ├── map_proteins_to_uniprot.py
│   │   ├── download_pdb_structures.py
│   │   └── generate_protein_graphs.py
│   ├── utils/
│   │   ├── init.py
│   │   ├── io_utils.py
│   │   ├── metrics.py
│   │   ├── molecule_utils.py
│   │   └── protein_utils.py
│   ├── training_workflow.py          # train_epoch, evaluate_epoch functions
│   └── train_model.py                # Main training script
├── weights/                          # Pre-trained model weights
│   ├── graphdta_davis_baseline_best_model.pt
│   ├── graphdta_kiba_baseline_best_model.pt
│   ├── graphdta3d_davis.pt

│   └── graphdta3d_kiba.pt

├── results/                          # Output JSON files with metrics
│   ├── graphdta_davis_results.json
│   ├── graphdta3d_davis_results.json
│   └── ...
├── .gitignore
├── environment.yml                   # For Conda
├── requirements.txt                  # For Pip
├── LICENSE
├── README.md
└── demo_predict.py                   # Script for making predictions with a trained model

## 🛠️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [Your_GitHub_Repo_Link]
    cd GraphDTA-3D
    ```

2.  **Create Environment & Install Dependencies:**

    * **Using Conda (recommended):**
        ```bash
        conda env create -f environment.yml
        conda activate graphdta3d_env
        ```
        *(Ensure your `environment.yml` specifies PyTorch, PyTorch Geometric, RDKit, Pandas, NumPy, scikit-learn, etc., matching the versions used in development, e.g., PyTorch 2.0.1+cu118).*

    * **Using Pip:**
        ```bash
        pip install -r requirements.txt
        ```
        *(Create `requirements.txt` listing necessary packages like `torch`, `torch-geometric`, `rdkit-pypi`, `pandas`, `numpy`, `scikit-learn`, `biopython`, `requests`, `tqdm`)*

3.  **Download External Data (if not included):**
    * Place raw Davis and KIBA dataset files into `data/davis/` and `data/kiba/` respectively.
    * Download UniProt Swiss-Prot FASTA file (`uniprot_sprot.fasta.gz`), unzip it, and place `uniprot_sprot.fasta` into `data/external/`.
        ```bash
        mkdir -p data/external
        wget -O data/external/uniprot_sprot.fasta.gz "[https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz)"
        gunzip data/external/uniprot_sprot.fasta.gz
        ```

## ⚙️ Usage

### 1. Data Preprocessing

Run the preprocessing scripts in the following order. These scripts will populate the `data/processed/` directory.

* **Generate Drug Graphs:**
    ```bash
    # For Davis
    python src/preprocessing/process_drug_graphs.py --dataset_name davis --csv_path ./data/davis/drugs.csv --smiles_col Canonical_SMILES --drug_id_col Drug_Index --output_dir ./data/processed/davis
    # For KIBA (assuming kiba_affinity_df.csv contains unique drug SMILES and their IDs)
    python src/preprocessing/process_drug_graphs.py --dataset_name kiba --csv_path ./data/kiba/kiba_affinity_df.csv --smiles_col SMILES --drug_id_col Drug_Index --output_dir ./data/processed/kiba
    ```

* **Process Protein Sequences (for GraphDTA baseline):**
    ```bash
    # For Davis
    python src/preprocessing/process_protein_sequences.py --dataset_name davis --csv_path ./data/davis/proteins.csv --sequence_col Sequence --protein_id_col Protein_Index --output_dir ./data/processed/davis
    # For KIBA
    python src/preprocessing/process_protein_sequences.py --dataset_name kiba --csv_path ./data/kiba/kiba_affinity_df.csv --sequence_col Sequence --protein_id_col Protein_Index --output_dir ./data/processed/kiba
    ```

* **Process Protein Structures (for GraphDTA-3D):**
    * First, create a FASTA file for your protein sequences (e.g., `davis_proteins.fasta`, `kiba_proteins.fasta`) and place it in `./data/{dataset_name}/`. The exact structure of this FASTA will depend on how your `proteins.csv` or `kiba_affinity_df.csv` is structured. The ID in the FASTA header should be usable for naming PDB files later (e.g., `>Protein_0`).
    * Map protein sequences to UniProt IDs:
        ```bash
        # For Davis (assuming davis_proteins.fasta exists)
        python src/preprocessing/map_proteins_to_uniprot.py --query_fasta ./data/davis/davis_proteins.fasta --uniprot_fasta ./data/external/uniprot_sprot.fasta --output_csv ./data/processed/davis/davis_uniprot_mapping.csv
        # For KIBA (assuming kiba_proteins.fasta exists)
        python src/preprocessing/map_proteins_to_uniprot.py --query_fasta ./data/kiba/kiba_proteins.fasta --uniprot_fasta ./data/external/uniprot_sprot.fasta --output_csv ./data/processed/kiba/kiba_uniprot_mapping.csv
        ```
    * Download PDB structures from AlphaFold:
        ```bash
        # For Davis
        python src/preprocessing/download_pdb_structures.py --mapping_csv ./data/processed/davis/davis_uniprot_mapping.csv --pdb_dir ./data/davis/pdb_files
        # For KIBA
        python src/preprocessing/download_pdb_structures.py --mapping_csv ./data/processed/kiba/kiba_uniprot_mapping.csv --pdb_dir ./data/kiba/pdb_files
        ```
    * Generate Protein Graphs from PDBs:
        ```bash
        # For Davis
        python src/preprocessing/generate_protein_graphs.py --pdb_dir ./data/davis/pdb_files --output_path ./data/processed/davis/davis_protein_graphs.pt
        # For KIBA
        python src/preprocessing/generate_protein_graphs.py --pdb_dir ./data/kiba/pdb_files --output_path ./data/processed/kiba/kiba_protein_graphs.pt
        ```

### 2. Model Training

Use the `src/train_model.py` script for training both GraphDTA and GraphDTA-3D models.

* **Train GraphDTA on Davis:**
    ```bash
    python src/train_model.py \
      --dataset_name davis \
      --model_architecture GraphDTA \
      --affinity_file drug_protein_affinity.csv \
      --affinity_col_name Affinity \
      --epochs 100 \
      --batch_size 512 \
      --learning_rate 0.0005 \
      --dropout_rate 0.2 \
      --output_dir ./results_and_weights/davis_graphdta_run1
    ```

* **Train GraphDTA-3D on Davis:**
    ```bash
    python src/train_model.py \
      --dataset_name davis \
      --model_architecture GraphDTA3D \
      --affinity_file drug_protein_affinity.csv \
      --affinity_col_name Affinity \
      --epochs 500 \
      --batch_size 512 \
      --learning_rate 0.0005 \
      --dropout_rate 0.3 \
      --output_dir ./results_and_weights/davis_graphdta3d_run1
    ```

* **Train GraphDTA on KIBA:**
    ```bash
    python src/train_model.py \
      --dataset_name kiba \
      --model_architecture GraphDTA \
      --affinity_file kiba_affinity_df.csv \
      --affinity_col_name KIBA_Score \
      --epochs 500 \
      --output_dir ./results_and_weights/kiba_graphdta_run1 
      # Add other params as needed
    ```

* **Train GraphDTA-3D on KIBA:**
    ```bash
    python src/train_model.py \
      --dataset_name kiba \
      --model_architecture GraphDTA3D \
      --affinity_file kiba_affinity_df.csv \
      --affinity_col_name KIBA_Score \
      --epochs 500 \
      --dropout_rate 0.3 \
      --output_dir ./results_and_weights/kiba_graphdta3d_run1
      # Add other params as needed
    ```
*(Adjust `--output_dir` to save weights and results for each run.)*

### 3. Prediction Demo

Use the `demo_predict.py` script to make predictions with a trained model.

* **Example for GraphDTA-3D (Davis model):**
    ```bash
    # Ensure you have a PDB file, e.g., data/davis/pdb_files/Protein_0.pdb
    python demo_predict.py \
      --model_architecture GraphDTA3D \
      --model_weights_path ./weights/graphdta3d_davis.pt \
      --smiles "DRUG_SMILES_STRING" \
      --protein_pdb_path ./data/davis/pdb_files/Protein_0.pdb \
      --dropout_rate 0.3
    ```

* **Example for GraphDTA (Davis baseline model):**
    ```bash
    python demo_predict.py \
      --model_architecture GraphDTA \
      --model_weights_path ./weights/graphdta_davis_baseline_best_model.pt \
      --smiles "DRUG_SMILES_STRING" \
      --protein_sequence "PROTEIN_SEQUENCE_STRING" \
      --dropout_rate 0.2
    ```

## 🗂️ Project Assets
* **Pre-trained Model Weights:** Located in the `weights/` directory (e.g., `graphdta3d_davis.pt`, `graphdta3d_kiba.pt`)[cite: 19].
* **Results Logs:** JSON files detailing performance metrics for each model and dataset are saved in the specified output directory during training (e.g., `results_and_weights/{run_name}/graphdta3d_davis_results.json`)[cite: 19].

## 🚀 Next Steps (Optional Future Work) [cite: 19]
* Deploy models for inference on larger compound libraries like ChEMBL for virtual screening.
* Incorporate pretrained protein language model embeddings (e.g., ESM, ProtT5) as additional node features for proteins.
* Experiment with more advanced Graph Neural Network (GNN) variants such as GIN or Graph Transformers.
* Develop an interactive web application (e.g., using Streamlit) for model demonstration.

## 📌 Conclusion
The GraphDTA-3D model, by integrating 3D structural information of proteins via AlphaFold-derived residue graphs, demonstrates a consistent improvement in DTA prediction accuracy over the sequence-only GraphDTA baseline[cite: 15, 20]. This work provides a reproducible and extensible framework for structure-aware DTA modeling, showing potential for applications in virtual screening and drug repurposing efforts[cite: 20].

## 📜 Citation
If you use this code or find this project helpful in your research, please consider citing:
```bibtex
@misc{YourName_GraphDTA3D_Year,
  author = {[Your Name]},
  title  = {GraphDTA-3D: Structure-Aware Drug–Target Binding Affinity Prediction},
  year   = {[Year_Of_Project_Completion]},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{[Link_To_Your_GitHub_Repo]}}
}