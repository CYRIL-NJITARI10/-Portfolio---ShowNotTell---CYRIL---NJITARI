# [Portofolio] - ShowNotTell - CYRIL - NJITARI - #5 - FROM POC TO PROD: StackOverflow Tag Prediction Service


## Introduction

"FROM POC TO PROD" is a Flask-based web application designed to predict tags for StackOverflow questions. Transitioning from a proof of concept (POC) to a production-ready application, this project encapsulates the transition phase where the focus is on building a reliable, scalable, and user-friendly application that leverages machine learning for text classification.

## Problem Description
In the realm of software development and data science Q&A, efficiently categorizing content is key to improving user experience and information retrieval. 
StackOverflow, a major platform in this field, requires a robust system that can automatically assign relevant tags to questions to ensure they are easily searchable and reach the right audience.

## Project Objectives

- Develop a Flask web application that provides an intuitive interface for users to input StackOverflow question titles and receive predicted tags.
- Implement a machine learning pipeline that processes text input and outputs prediction tags, leveraging trained models and state-of-the-art text embedding techniques.
- Transition the project from a proof of concept to a production-grade system with an emphasis on code quality, maintainability, and scalability.
- Ensure that the application is secure, with appropriate error handling and validation to mitigate potential risks associated with web deployments.

## Key Features

- User-Friendly Interface: A Bootstrap-based front end that is simple and intuitive for end-users to interact with the machine learning model.
- Machine Learning Integration: Integration of TensorFlow and Keras for running a neural network that predicts tags based on the text of a StackOverflow question.
- Scalability: Structured codebase that is ready for scaling up to handle increased load and additional features as needed.
- Robust Testing: Comprehensive test suite for the prediction logic, ensuring reliability and accuracy of the tag predictions.

## How to Use the Service
- Start the Application: Run the Flask app to start the web service.
- Access the Interface: Open the provided URL in a web browser to access the prediction interface.
- Input and Predict: Type in a StackOverflow question title and submit to receive the predicted tags.

## Installation and Setup

- Ensure that you have Python and Flask installed.
- Clone the repository and navigate to the predict directory.
- Install required Python packages using pip install -r requirements.txt.
- Run python app.py to start the Flask server.
- Open a web browser and navigate to http://127.0.0.1:5000/ to access the application.

## Testing
- Navigate to the tests directory to view and run the test suite.
- Use the command python -m unittest discover to run all tests.

## Technologies
- Flask: Web framework for building the web service.
- TensorFlow and Keras: For loading the machine learning model and making predictions.
- Bootstrap: For styling the frontend web interface.
- Pandas: For data manipulation during preprocessing.
- NumPy: For numerical operations on data.
- BERT: For text embedding to transform question titles into vector representations.

## Project Structure
- app.py: Flask application initialization and route definitions.
- run.py: Core prediction logic and machine learning model handling.
- tests/: Directory containing unit tests for various components of the project.

## Future Enhancements
- Integrate with a CI/CD pipeline for automated testing and deployment.
- Add more sophisticated error handling and input validation to the frontend.
- Explore optimization opportunities for the machine learning model to improve prediction speed and accuracy.

## Conclusion
"FROM POC TO PROD" exemplifies a successful transition from a machine learning prototype to a fully-fledged predictive service. This service not only demonstrates the power of natural language processing but also sets a blueprint for deploying machine learning models in production environments.