import csv
import argparse

def read_csv_data(file_path):
    """
    Reads CSV data from the given file path.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        tuple: A tuple containing two lists: hourly_rainfall_data and soil_rainfall_data.
               Returns (None, None) if an error occurs.
    """
    hourly_rainfall_data = []
    soil_rainfall_data = []
    try:
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            if 'hourly_rainfall' not in reader.fieldnames or 'soil_rainfall' not in reader.fieldnames:
                print(f"Error: CSV file must contain 'hourly_rainfall' and 'soil_rainfall' columns.")
                return None, None
            for row_number, row in enumerate(reader, start=2):  # Start from 2 to account for header
                try:
                    hourly_rainfall = float(row['hourly_rainfall'])
                    soil_rainfall = float(row['soil_rainfall'])
                    hourly_rainfall_data.append(hourly_rainfall)
                    soil_rainfall_data.append(soil_rainfall)
                except ValueError:
                    print(f"Warning: Could not convert data to float in row {row_number}. Skipping row.")
                except TypeError: # Handles cases where a row might be None or not a dict
                    print(f"Warning: Invalid row format in row {row_number}. Skipping row.")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None
    return hourly_rainfall_data, soil_rainfall_data

def calculate_correlation(data1, data2):
    """
    Calculates the Pearson correlation coefficient between two datasets.

    Args:
        data1 (list): A list of numerical data.
        data2 (list): A list of numerical data.

    Returns:
        float: The Pearson correlation coefficient, or None if calculation is not possible.
    """
    if not data1 or not data2:
        print("Error: Input data lists cannot be empty.")
        return None
    if len(data1) != len(data2):
        print("Error: Data lists must have the same length.")
        return None
    if len(data1) < 2: # Correlation is not well-defined for less than 2 data points
        print("Warning: Correlation is not well-defined for less than 2 data points.")
        return None

    n = len(data1)
    sum_x = sum(data1)
    sum_y = sum(data2)
    sum_x_sq = sum(x**2 for x in data1)
    sum_y_sq = sum(y**2 for y in data2)
    sum_xy = sum(x * y for x, y in zip(data1, data2))

    numerator = n * sum_xy - sum_x * sum_y
    denominator_x = (n * sum_x_sq - sum_x**2)**0.5
    denominator_y = (n * sum_y_sq - sum_y**2)**0.5

    if denominator_x == 0 or denominator_y == 0:
        # This can happen if all values in one of the datasets are the same.
        print("Warning: Cannot calculate correlation, standard deviation of one or both datasets is zero.")
        return None

    correlation = numerator / (denominator_x * denominator_y)
    return correlation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate correlation between hourly rainfall and soil rainfall from a CSV file.")
    parser.add_argument("csv_file", help="Path to the CSV file containing rainfall data.")
    args = parser.parse_args()

    hourly_data, soil_data = read_csv_data(args.csv_file)

    if hourly_data is not None and soil_data is not None:
        print("Successfully loaded data:")
        # Print a sample of the data
        print("Hourly Rainfall (first 5 entries):", hourly_data[:5])
        print("Soil Rainfall (first 5 entries):", soil_data[:5])

        correlation_coefficient = calculate_correlation(hourly_data, soil_data)
        if correlation_coefficient is not None:
            print(f"\nPearson Correlation Coefficient: {correlation_coefficient:.4f}")
