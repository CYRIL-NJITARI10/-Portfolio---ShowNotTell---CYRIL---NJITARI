# [Portofolio] - ShowNotTell - CYRIL - NJITARI - #4 - Creating an API with FastAPI for Iris Dataset Prediction

## Introduction
This project showcases a FastAPI application that serves a machine learning model for making predictions on the Iris Flower dataset. We have created a RESTful API that encapsulates data processing and machine learning logic in a way that is accessible and easy to use.

In many sectors, the ability to quickly and accurately predict outcomes from complex datasets is a valuable asset. 
Particularly in the domain of botany, where classifying species based on measurements can be a tedious and error-prone task, machine learning models offer a significant improvement in efficiency and accuracy. 
This project was conceived to address the need for a reliable, scalable, and accessible method to classify Iris flowers based on their physical characteristics. 
The Iris Flower dataset, a foundational dataset in machine learning for pattern recognition, was chosen as the basis for this project. 
However, despite the dataset's simplicity, the principles and practices applied here are extendable to more complex and larger scale predictive tasks.

## Project Objectives
- Create a RESTful API that allows users to interact with a machine learning model in a stateless manner via HTTP requests.
- Provide a seamless data ingestion process that takes raw data from the Iris Flower dataset and prepares it for analysis.
- Implement data preprocessing, transformation, and splitting to facilitate effective model training and evaluation.
- Utilize a classification model from the Scikit-Learn library to predict Iris species based on their characteristics.
- Construct a workflow that encompasses the end-to-end process of machine learning — from data acquisition to model prediction.
- Ensure that the API is well-documented, user-friendly, and adheres to best practices, making it accessible for developers and users with varying levels of technical expertise.
- Integrate Google Firestore to manage model parameters dynamically, allowing for real-time updates and modifications.
- Provide robust endpoints for model training, prediction, and parameter modification, enabling the model to improve and adapt over time.
- Automate and simplify complex machine learning tasks, making predictive analytics more accessible to a wider audience.
- Demonstrate the capabilities of FastAPI in creating efficient and high-performing machine learning applications.

These objectives are aimed at not only solving the immediate classification problem but also demonstrating a scalable approach to deploying machine learning models in production environments.

## RESTful API Concepts
- Resources: Unique entities that can be manipulated using endpoints identified by URIs.
- HTTP Methods: Standard methods such as GET, POST, PUT, and DELETE are used for CRUD operations.
- Statelessness: No client context is stored on the server between requests.

## FastAPI Features
- Endpoint: A specific path accessed via the API to perform operations on resources.
- HTTP Methods: Employ GET for retrieving, POST for creating, PUT for updating, and DELETE for deleting resources.
- Request and Response: Data sent to and from the API in the form of JSON or form data.

## Project Completion Steps

### Step 1: Installation
The required libraries have been installed and are listed in the requirements.txt file.

### Step 2: Launching the Application
The application has been successfully launched and is running. It can be accessed at the root endpoint.

### Step 3: API Root Redirection
The root endpoint of the API has been redirected to the automatically generated Swagger documentation.

### Step 4: Swagger Documentation
The Swagger documentation, detailing all the available endpoints and their functionalities, can be accessed to interact with the API.

### Step 5: API Interaction
The /hello route has been tested to ensure correct API functioning using Swagger UI, and tools like Insomnia/Postman.

### Step 6: Dataset Access
An endpoint has been created to facilitate the downloading and storage of the Iris dataset within the src/data folder.

### Step 7: Dataset Loading
An endpoint is available to load the Iris dataset file into a dataframe and return the data as a JSON response.

### Step 8: Data Processing
We have added an endpoint that allows for the preprocessing of the Iris dataset to prepare it for model training.

### Step 9: Dataset Splitting
The API provides an endpoint to split the dataset into training and testing sets, which can be retrieved in JSON format.

### Step 10: Model Parameters Configuration
The classification model parameters are configured and stored in src/config/model_parameters.json.

### Step 11: Model Training
The API includes an endpoint for training the classification model with the processed dataset, with the trained model saved in src/models.

### Step 12: Prediction Endpoint
We have implemented an endpoint to make predictions using the trained model, returning predictions as a JSON response.

### Step 13: Firestore Integration
A Firestore collection named "parameters" has been established for managing model parameters.

### Step 14: Firestore Parameter Retrieval
There is an endpoint in place to fetch parameters from the Firestore collection.

### Step 15: Firestore Parameter Updating
Endpoints have been added to the API for updating existing parameters or adding new ones to the Firestore collection.

## Documentation and Resources
- FastAPI: FastAPI Documentation
- Google Cloud Firestore: Firestore Python Client Library
- Scikit-Learn: Scikit-Learn User Guide
- Pandas: Pandas Documentation

## Project Setup and Execution
Instructions on how to set up the project environment, execute the data pipeline, and deploy the services.


Coming soon 

coming soon

## Visualization 
how Metabase is utilized for creating dashboards that deliver key business insights from the processed data.

coming soon

## Monitoring, Alerting, and Reliability
system monitoring strategy, alerting mechanisms, and how reliability is ensured across the pipeline.


coming soon

## Security Measures
security protocols implemented for data protection.


coming soon


## Challenges and Solutions


coming soon 

## Conclusion and Future Work

The API is fully functional, with endpoints thoroughly tested and documented. 
The machine learning workflow from dataset acquisition to model predictions is operational and ready for use.

Next ... coming soon




