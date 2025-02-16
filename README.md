# **MedDataConverter**
*A Python module for converting medical imaging datasets (`.mha`, `.raw`, `.dcm`) to `.pickle` for machine learning workflows.*

---

## **1️⃣ Clone the Repository**

```bash
git clone https://github.com/rasmushy/MedDataConverter.git
cd MedDataConverter
```

---

## **2️⃣ Set Up the Conda Environment**
Run the following command to create and activate the `meddataconverter` environment:

```bash
conda env create -f environment.yml
conda activate meddataconverter
```

This will install **all necessary dependencies**.

---

## **3️⃣ Install TIGRE (Tomographic Iterative GPU-based Reconstruction)**
Since **TIGRE** is not available via `conda`, install it manually:

```bash
git clone --branch 2.3 https://github.com/CERN/TIGRE.git TIGRE-2.3
cd TIGRE-2.3/Python
python setup.py install
```

### **🔹 Troubleshooting TIGRE**
If you encounter errors:
- Ensure **GCC** and **NVCC (CUDA Compiler)** are installed.
- If you see a **NumPy version mismatch error**, reinstall NumPy:

  ```bash
  pip install --force-reinstall numpy
  ```

- Verify the installation:

  ```bash
  python -c "import tigre; print('TIGRE installed successfully!')"
  ```

---

## **4️⃣ Optional: Install MedDataConverter as a Package**
Inside the `MedDataConverter` directory, install the module:

```bash
pip install -e .
```

---

## **5️⃣ Convert `.mha` or `.raw` to `.pickle`**
Run the dataset processor (make sure u are using conda env: conda activate meddataconverter)

```bash
python src/datasetProcessor.py
```

Follow the interactive prompts to:
- Select a dataset (`.mha` / `.raw`)
- Define configurations
- Generate `.mat` and `.pickle` files

Example:

```bash
Enter the dataset name (e.g., 'head', 'leg', 'toothfairycbct'): tooth
Select a file by number: 2
Selected file is .mha. Shape and dtype will be extracted automatically.
```

---

## **6️⃣ Verify the Output**
Check the output folder for `.mat` and `.pickle` files:

```bash
ls ./mat_data/<dataset_name>/
ls ./pickle_data/<dataset_name>/
```
---
