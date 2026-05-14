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

```bash
git clone https://github.com/imosudi/unsupervised-ml-kmeans-analysis.git
cd unsupervised-ml-kmeans-analysis

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


## License

This project is licensed under the **MIT license**. See [LICENSE](./LICENSE) for details.

## Author

[![GitHub](https://img.shields.io/badge/GitHub-imosudi-181717?logo=github)](https://github.com/imosudi)
[![Email](https://img.shields.io/badge/Email-Contact-blue?logo=gmail)](mailto:mosudi.isiaka@gmail.com)