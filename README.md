# solid_principles_python

This Python script demonstrates a data processing pipeline that was refactored to adhere to the SOLID principles, focusing on the Single Responsibility Principle (SRP), Dependency Inversion Principle (DIP), and a basic form of the Interface Segregation Principle (ISP).

## Overview

The original script lacked adherence to SOLID principles and was refactored to improve maintainability and extensibility. It now defines a `DataProcessor` interface with methods for input and output columns, along with implementations for standardizing, encoding, and filling NaN values in a pandas DataFrame. The `DataPipeline` class orchestrates the data processing steps by applying each processor sequentially.

## SOLID Principles Demonstrated

- **Single Responsibility Principle (SRP):** Each `DataProcessor` implementation has a single responsibility related to processing a specific aspect of the data.
- **Open/Closed Principle (OCP):** The `DataPipeline` class is open for extension but closed for modification, allowing new processors to be added without changing existing code.
- **Liskov Substitution Principle (LSP):** Not explicitly demonstrated in this code.
- **Interface Segregation Principle (ISP):** The `DataProcessor` interface is focused and does not force implementations to depend on unnecessary methods.

## Code for Individual SOLID Principles

The `solid_principle` folder contains the code examples for each individual SOLID principle, demonstrating how the principles are applied in isolation.

## File Structure
data_pipeline.py: Contains the main script with the DataProcessor, processor implementations, and DataPipeline class.
data/: Directory containing the sample dataset (data.parquet) and the output preprocessed data (preprocessed_data.parquet).
solid_principle/: Directory containing code examples for each individual SOLID principle.

## License
This project is licensed under the MIT License - see the LICENSE file for details.


Feel free to customize this as needed for your project!
