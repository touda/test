import unittest
import os
import csv
import io
from unittest.mock import patch # For mocking print and stderr

# Import functions from the script to be tested
from rainfall_correlation import read_csv_data, calculate_correlation

class TestRainfallCorrelation(unittest.TestCase):

    def create_dummy_csv(self, data, filename="dummy.csv"):
        """Helper function to create a temporary CSV file for testing."""
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)
        return filename

    def tearDown(self):
        """Clean up any created files after tests."""
        if os.path.exists("dummy.csv"):
            os.remove("dummy.csv")
        if os.path.exists("non_existent_file.csv"): # Should not exist, but good practice
            pass
        if os.path.exists("value_error.csv"):
            os.remove("value_error.csv")

    def test_read_csv_data_success(self):
        csv_data = [
            ["hourly_rainfall", "soil_rainfall"],
            ["1.0", "0.5"],
            ["2.0", "1.5"],
            ["3.0", "2.5"]
        ]
        dummy_file = self.create_dummy_csv(csv_data, "dummy.csv")
        hourly, soil = read_csv_data(dummy_file)
        self.assertEqual(hourly, [1.0, 2.0, 3.0])
        self.assertEqual(soil, [0.5, 1.5, 2.5])

    @patch('builtins.print') # Mock print to check its output
    def test_read_csv_data_file_not_found(self, mock_print):
        hourly, soil = read_csv_data("non_existent_file.csv")
        self.assertIsNone(hourly)
        self.assertIsNone(soil)
        # Check if the specific error message for file not found was printed
        mock_print.assert_any_call("Error: File not found at non_existent_file.csv")


    @patch('builtins.print') # Mock print to check its output
    def test_read_csv_data_value_error(self, mock_print):
        csv_data = [
            ["hourly_rainfall", "soil_rainfall"],
            ["1.0", "0.5"],
            ["abc", "1.5"], # Invalid data
            ["3.0", "xyz"], # Invalid data
            ["4.0", "3.5"]
        ]
        value_error_file = self.create_dummy_csv(csv_data, "value_error.csv")
        hourly, soil = read_csv_data(value_error_file)
        self.assertEqual(hourly, [1.0, 4.0])
        self.assertEqual(soil, [0.5, 3.5])
        # Check if warning messages were printed for skipped rows
        mock_print.assert_any_call("Warning: Could not convert data to float in row 3. Skipping row.")
        mock_print.assert_any_call("Warning: Could not convert data to float in row 4. Skipping row.")

    @patch('builtins.print')
    def test_read_csv_data_missing_columns(self, mock_print):
        csv_data = [
            ["header1", "header2"],
            ["1.0", "0.5"]
        ]
        dummy_file = self.create_dummy_csv(csv_data, "dummy.csv")
        hourly, soil = read_csv_data(dummy_file)
        self.assertIsNone(hourly)
        self.assertIsNone(soil)
        mock_print.assert_any_call("Error: CSV file must contain 'hourly_rainfall' and 'soil_rainfall' columns.")


    def test_calculate_correlation_success(self):
        # Perfect positive correlation
        data_pos_h = [1, 2, 3, 4, 5]
        data_pos_s = [2, 4, 6, 8, 10]
        self.assertAlmostEqual(calculate_correlation(data_pos_h, data_pos_s), 1.0)

        # Perfect negative correlation
        data_neg_h = [1, 2, 3, 4, 5]
        data_neg_s = [5, 4, 3, 2, 1]
        self.assertAlmostEqual(calculate_correlation(data_neg_h, data_neg_s), -1.0)

        # No correlation (or very low)
        # Note: True zero correlation is hard to achieve with small datasets without specific construction
        # For this example, let's use data that should result in a value close to zero.
        data_zero_h = [1, 2, 3, 4, 5, 6]
        data_zero_s = [2, 5, 3, 6, 1, 4] # Somewhat random
        # We expect a value, not necessarily exactly 0 for small random-like sets.
        # Let's check a known case for near zero.
        # X = [1, 2, 3, 4, 5], Y = [2, 1, 0, 1, 2] (symmetric around middle) -> should be 0
        data_zero_h_2 = [1,2,3,4,5]
        data_zero_s_2 = [2,1,0,1,2]
        self.assertAlmostEqual(calculate_correlation(data_zero_h_2, data_zero_s_2), 0.0)

        # General case
        data_gen_h = [10, 12, 15, 11, 13]
        data_gen_s = [3, 4, 5, 3.5, 4.5] # Expected positive correlation
        # Manual calculation for this specific case:
        # n=5
        # sum_x = 61, sum_y = 20
        # sum_x_sq = 759, sum_y_sq = 84.5
        # sum_xy = 249
        # num = 5*249 - 61*20 = 1245 - 1220 = 25
        # den_x_sq = 5*759 - 61^2 = 3795 - 3721 = 74
        # den_y_sq = 5*84.5 - 20^2 = 422.5 - 400 = 22.5
        # corr = 25 / (sqrt(74) * sqrt(22.5)) = 25 / (8.6023 * 4.7434) = 25 / 40.805 = 0.61266
        self.assertAlmostEqual(calculate_correlation(data_gen_h, data_gen_s), 0.61266, places=5)

    @patch('builtins.print')
    def test_calculate_correlation_empty_lists(self, mock_print):
        self.assertIsNone(calculate_correlation([], []))
        mock_print.assert_any_call("Error: Input data lists cannot be empty.")
        self.assertIsNone(calculate_correlation([1,2], []))
        mock_print.assert_any_call("Error: Input data lists cannot be empty.")
        self.assertIsNone(calculate_correlation([], [1,2]))
        mock_print.assert_any_call("Error: Input data lists cannot be empty.")

    @patch('builtins.print')
    def test_calculate_correlation_different_lengths(self, mock_print):
        self.assertIsNone(calculate_correlation([1, 2, 3], [1, 2]))
        mock_print.assert_any_call("Error: Data lists must have the same length.")

    @patch('builtins.print')
    def test_calculate_correlation_insufficient_data(self, mock_print):
        self.assertIsNone(calculate_correlation([1], [2]))
        mock_print.assert_any_call("Warning: Correlation is not well-defined for less than 2 data points.")

    @patch('builtins.print')
    def test_calculate_correlation_zero_std_dev(self, mock_print):
        # Case 1: First list has zero standard deviation
        self.assertIsNone(calculate_correlation([2, 2, 2, 2], [1, 2, 3, 4]))
        mock_print.assert_any_call("Warning: Cannot calculate correlation, standard deviation of one or both datasets is zero.")
        
        # Case 2: Second list has zero standard deviation
        self.assertIsNone(calculate_correlation([1, 2, 3, 4], [5, 5, 5, 5]))
        mock_print.assert_any_call("Warning: Cannot calculate correlation, standard deviation of one or both datasets is zero.")

        # Case 3: Both lists have zero standard deviation
        self.assertIsNone(calculate_correlation([7, 7, 7], [8, 8, 8]))
        mock_print.assert_any_call("Warning: Cannot calculate correlation, standard deviation of one or both datasets is zero.")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    # Using exit=False to prevent sys.exit, which can be an issue in some environments
    # The first-arg-is-ignored is also a common practice when running unittest.main from a script
    # Alternatively, for command-line execution: python -m unittest test_rainfall_correlation.py
