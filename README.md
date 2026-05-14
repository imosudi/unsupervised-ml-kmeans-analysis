# Unsupervised ML - K-Means Clustering Analysis

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)
![NumPy](https://img.shields.io/badge/NumPy-Scientific_Computing-013243?logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualisation-green)
![K-Means](https://img.shields.io/badge/K--Means-Clustering-red)
![Silhouette Score](https://img.shields.io/badge/Silhouette-Score-purple)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Completed-success)

This project explores unsupervised machine learning using the K-Means clustering algorithm on synthetic datasets generated with scikit-learn. The analysis includes cluster visualisation, silhouette score evaluation, and parameter sensitivity experiments involving cluster count and random seed variation.

## Deployment

Clone the repository and run the notebook or Python script locally.

### Linux/Mac

```bash
git clone https://github.com/imosudi/unsupervised-ml-kmeans-analysis.git
cd unsupervised-ml-kmeans-analysis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows

```bash
git clone https://github.com/imosudi/unsupervised-ml-kmeans-analysis.git
cd unsupervised-ml-kmeans-analysis
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Run:

```bash
python main.py

```

Or launch Jupyter Notebook:
```bash
jupyter notebook

```


## Project goals

- Generate synthetic datasets using `sklearn.datasets.make_classification`
- Apply K-Means clustering on multi-class datasets
- Visualise clustering regions using meshgrid decision boundaries
- Evaluate clustering quality using silhouette scores
- Analyse the impact of:
  - Different cluster counts (`k`)
  - Different random seeds
- Compare clustering stability and performance
- Strengthen practical understanding of unsupervised machine learning

## Project Structure

```
unsupervised-ml-kmeans-analysis/
├── assets/
│   └── screenshots/          # Screenshots for the analysis report
│       ├── task1.png
│       ├── task3.png
│       ├── task5_1.png
│       └── task5_2.png
├── doc/
│   ├── assets/
│   │   └── screenshots/      # Copies of screenshots for documentation
│   ├── generate_pdf.py       # Script to generate PDF from markdown
│   └── Unsupervised_ML_Analysis.md  # Detailed analysis report
├── LICENSE
├── main.ipynb               # Jupyter notebook with the analysis
├── main.py                  # Python script version
├── README.md                # This file
└── requirements.txt         # Python dependencies
```

## Documentation

The detailed analysis report is available in `doc/Unsupervised_ML_Analysis.md`. To generate a PDF version:

```bash
cd doc
python generate_pdf.py
```

This will create `Unsupervised_ML_Analysis.pdf` in the `doc/` directory.

This project is licensed under the **MIT license**. See [LICENSE](./LICENSE) for details.

## Author


**Mosudi Isiaka O.**  
![Email](https://img.shields.io/badge/Email-Contact-blue?logo=gmail) [mosudi.isiaka@gmail.com](mailto:mosudi.isiaka@gmail.com)  
![GitHub](https://img.shields.io/badge/GitHub-imosudi-181717?logo=github) [https://github.com/imosudi](https://github.com/imosudi)

