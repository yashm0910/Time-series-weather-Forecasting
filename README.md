# Next-Hour Weather Forecasting using Transformers

A multivariate time-series deep learning model with an interactive Streamlit dashboard for predicting weather conditions one hour ahead.

## Project Overview

This project uses a Transformer Encoder architecture to forecast next-hour weather conditions based on the previous 24 hours of data. Unlike traditional forecasting methods like ARIMA or LSTM networks, this approach leverages self-attention mechanisms to capture complex interactions between multiple weather features over time.

The model predicts five key weather parameters:
- Humidity percentage
- Temperature in Celsius (actual and apparent)
- Atmospheric pressure in millibars
- Wind speed in kilometers per hour
- Rain likelihood probability

## What Makes This Different

Traditional weather forecasting models often struggle with capturing relationships between multiple weather variables simultaneously. This project addresses that by using a Transformer architecture originally designed for natural language processing, adapted here for time-series weather data. The self-attention mechanism allows the model to weigh the importance of different time steps and features when making predictions.

## Project Structure

The project is organized into separate notebooks for better workflow management:

- **preprocessing** - Handles data cleaning, missing value treatment, and feature engineering
- **model prep and window generation** - Creates sliding window sequences (24 hours of input to predict the next hour)
- **transformer training and evaluation** - Contains the model architecture, training loop, and model checkpointing and Performance analysis with metrics and visualization

The saved model artifacts are stored in a dedicated folder:
- Trained model weights
- Feature scaler for normalization
- Feature and target column configurations

The processed data and Streamlit application file complete the structure.

## How the Model Works

The architecture follows this pipeline:

1. Takes 24 hours of historical weather data as input
2. Embeds the features into a higher-dimensional space
3. Adds positional encoding to maintain temporal order
4. Processes through multiple Transformer Encoder layers
5. Applies mean pooling across the sequence
6. Outputs predictions through a regression head

Each input consists of multiple weather features across 24 time steps. The model learns to identify patterns and relationships that indicate what the weather will be like in the next hour.

## Model Performance

The model was evaluated on a held-out test set with the following results:

- Humidity: MAE of 0.0327, RMSE of 0.0456
- Temperature: MAE of 0.0129, RMSE of 0.0164
- Apparent Temperature: MAE of 0.0147, RMSE of 0.0189
- Pressure: MAE of 0.0089, RMSE of 0.0332
- Wind Speed: MAE of 0.0372, RMSE of 0.0511

These metrics indicate stable and reliable forecasting across all weather variables. The humidity predictions show particularly strong correlation with actual conditions, making the rain likelihood estimates quite dependable.

## Running the Application

Install the required dependencies first, then launch the Streamlit dashboard. The application automatically loads the most recent 24 hours from the processed dataset and displays:

- Next hour weather predictions
- Historical humidity trends
- Rain probability assessment
- Natural language weather summary

The interface is designed to be intuitive for users without technical backgrounds while still providing detailed information for those interested in the underlying data.

## Future Development

Several enhancements are planned for future versions:

Multi-step forecasting to predict conditions 6 to 12 hours ahead would provide more comprehensive weather planning. Integrating additional features like cloud cover and dew point could improve prediction accuracy. A hybrid architecture combining the Transformer with classification heads could predict specific rain types. Real-time API integration would enable live weather monitoring and reporting.

## Technical Insights

The Transformer architecture proved particularly effective for this task because weather patterns involve complex, non-linear relationships between variables. Temperature affects humidity, pressure influences wind patterns, and these relationships vary across different time scales. The attention mechanism naturally captures these dynamics without manual feature engineering.

One interesting finding was that the model performs better during stable weather conditions and shows larger errors during rapid transitions. This suggests that incorporating rate-of-change features or ensemble methods might further improve performance during volatile periods.

The project demonstrates that modern NLP architectures can be successfully adapted for time-series forecasting tasks, especially when dealing with multivariate data where feature interactions matter.
