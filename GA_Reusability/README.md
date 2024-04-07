# [Portofolio] - ShowNotTell - CYRIL - NJITARI - #2 -  GA_Reuseability

## Introduction
GA_Reuseability is an advanced software package that provides a flexible, extensible framework for solving optimization problems using genetic algorithms. It is designed to be adaptable for a wide range of problems, such as the classic Traveling Salesperson Problem (TSP) and games like Mastermind, showcasing the power of Object-Oriented Programming (OOP) principles in creating reusable and maintainable code.

## Problem Description
Optimization problems, including but not limited to route optimization and strategy games, pose significant challenges in both academic and real-world contexts. Solving these problems often requires sophisticated algorithms that can adapt to various constraints and objectives. GA_Reuseability aims to provide a generic solution for such problems, leveraging the adaptability of genetic algorithms.

## Project Objectives
Create a modular genetic algorithm framework that can be applied to different optimization problems without substantial changes to the core logic.
Demonstrate the application of OOP concepts like inheritance, encapsulation, and polymorphism to build a flexible software architecture.
Implement concrete problem classes for TSP and Mastermind, showcasing the reusability of the base genetic algorithm classes.
Develop a set of robust unit tests to validate the functionality and reliability of the genetic algorithms and ensure the quality of the solution.
Ensure that the solution is efficient, with a focus on the performance and accuracy of the genetic algorithms.

## Key Features

Generic Genetic Algorithm Solver: A core solver class that implements the genetic algorithm and can be extended for various problems.
Problem-Specific Implementations: Specialized classes for TSP and Mastermind that inherit from a base GAProblem class.
Customizable Selection and Mutation Rates: Ability to configure genetic algorithm parameters like selection rate and mutation rate.
Evolution Monitoring: Methods to observe the evolution process, including the best individual per generation and a summary of the population's fitness.
Visualization: For TSP, a plotting function to visualize the cities and the current best route.
Automated Testing: A comprehensive test suite to ensure each part of the framework functions as intended.

## Usage
Configure the genetic algorithm parameters in the genetic_spec.py module.
Instantiate the GASolver with the problem you wish to solve, such as MastermindProblem or TSProblem.
Call evolve_until() method on the solver instance to start the evolutionary process.
Use get_best_individual() to retrieve the best solution after the evolution.

## Installation and Setup
Ensure Python 3.x is installed.
Clone the repository to your local machine.
No external dependencies are required, as the framework uses Python's standard libraries.

## Testing
Navigate to the tests directory to view and run the test suite.
Execute python -m unittest to run all tests and validate the correctness of the implementation.