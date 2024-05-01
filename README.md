# Evaluating the Adaptability and Power of Language Models for Time Series Forecasting

This project explores the application of advanced Language Models (LMs), specifically within the financial domain, leveraging the Financial News and Stock Price Integration Dataset (FNSPID). This dataset uniquely combines numerical and textual data, offering a multimodal approach to time series forecasting. 

We assess the capabilities of language models (LMs) in a fine-tuned setting against traditional statistical, machine learning, and deep learning models.

Our approach incorporates textual financial news features with quantitative financial indicators to predict market movements, characterized by BUY, SELL, or HOLD decisions. We employ ChronosT5, an innovative framework that merges the predictive power of pretrained probabilistic time series models with the T5 family of language models. The results demonstrate ChronosT5's effectiveness in zero-shot forecasting tasks, highlighting its potential as a powerful component in financial forecasting pipelines.


## Project Structure
`models/`
  - Various Python notebooks containing the implementations of the ChronosT5, DeepAR, GARCH, and XGBoost models.

`utils/`
  - `data_processing.py`: Functions for processing data, including generating technical financial signals and golden labels for model training.
  - `evaluation.py`: Functions for running market simulation to evaluate the performance of the trained models.

`data/`
  - Processed AAPL data in CSV format from 2001-01-01 to 2019-12-31.
  - Contains technical signals which include RSI, CCI, PPO, and BB%

## Getting Started
Prerequisites
- Python 3.8 or above
- Jupyter Notebook or Google Collab
- Libraries: pandas, numpy, matplotlib, scikit-learn, statsmodels, xgboost, pytorch (for DeepAR and ChronosT5)

## Installation 
- Clone the repository: `git clone https://github.com/ManiMSA/NLP_Group_Project_LLMTS.git`
- Install required Python packages: `pip install -r requirements.txt`
