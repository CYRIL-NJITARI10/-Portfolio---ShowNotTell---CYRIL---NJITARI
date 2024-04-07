# [Portfolio] - ShowNotTell - CYRIL - NJITARI - #3 - Police Department Crime Analytics - GCP

## Context and Problem

The local police department's analytics division aims to gain insights into crime patterns within their jurisdiction. A dataset encompassing all recorded crimes has been provided, and as a Data Engineer, my role is to analyze this data to answer critical questions posed by the department, facilitating better resource allocation and crime prevention strategies.

## Project Objectives

- Modify the dataset to reflect the correct timeline by adding 3 years to all dates.
- Analyze monthly crime trends over the past 5 years.
- Identify the top 10 locations for theft occurrences for each of the past 3 years.
- Determine whether the overall crime rate is increasing or decreasing.
- Establish which areas are safest during the late-night hours (10 PM to 4 AM).
- Identify which types of crimes have the highest arrest rates between 2016 and 2019.
- Manipulate GCP tools

## Project Architecture and Workflow:

This project utilizes two distinct architectural designs for processing and analyzing crime data.

### ARCHITECTURE 1

![img.png](img.png)
### ARCHITECTURE 2
![img_1.png](img_1.png)

## System Design

1. Data Ingestion and Processing:
- Cloud Functions: Trigger data processing jobs and perform initial data transformation.
- Cloud Dataproc: A managed Spark and Hadoop service that processes large datasets.
- Cloud Storage: Stores processed datasets and intermediate results.
- BigQuery: Used for running SQL queries on large datasets to answer the research questions.

2. Architecture Specific Steps:
- Architecture 1:

Data is processed by a temporary Spark cluster via Cloud Dataproc and stored in Cloud Storage as parquet files.
Further processed by Cloud Functions to create BigQuery tables for analysis.
Optional: Cloud Functions are used to generate visualizations stored in Cloud Storage.
- Architecture 2:

A similar process as Architecture 1 with a direct transfer of processed data into BigQuery for analysis.

Pub/Sub: To facilitate event-driven processing and decouple data sources from consumers.
Cloud Scheduler: To orchestrate and schedule jobs in a cron-like fashion.

Technologies Used:
- GCP Cloud Functions: For serverless event-driven execution.
- GCP Cloud Dataproc: To handle Spark jobs.
- GCP BigQuery: For data warehousing and analytics.
- Terraform: For infrastructure as code management.

## Project Setup and Execution
Instructions on how to set up the project environment, execute the data pipeline, and deploy the services.


## Data Transformation and Analysis
Explanation of the data transformation logic, SQL queries used for analysis, and the methodology for trend analysis.

Coming soon

## Visualization and Reporting
how the results are presented to the stakeholders, including any dashboards or reports.

Coming soon

## Monitoring and Optimization
how the pipeline is monitored for performance and what optimization techniques are applied to ensure efficient data processing.

Coming soon

## Security and Compliance
security measures and compliance protocols implemented to safeguard sensitive data.

Coming soon


## Challenges and Solutions


potential enhancements, such as predictive analytics models to forecast crime trends or integrating additional datasets for deeper insights.

Coming soon

## Conclusion and Future Work

Coming soon
