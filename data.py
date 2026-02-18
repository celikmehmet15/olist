from pathlib import Path
import pandas as pd


class Olist:
    """
    The Olist class provides methods to interact with Olist's e-commerce data.

    Methods:
        get_data():
            Loads and returns a dictionary where keys are dataset names (e.g., 'sellers', 'orders')
            and values are pandas DataFrames loaded from corresponding CSV files.

        ping():
            Prints "pong" to confirm the method is callable.
    """
    def get_data(self):
        """
        This function returns a Python dict.
        Its keys should be 'sellers', 'orders', 'order_items' etc...
        Its values should be pandas.DataFrames loaded from csv files
        """
    def get_data(self):
        """
        This function returns a Python dict.
        Its keys should be 'sellers', 'orders', 'order_items' etc...
        Its values should be pandas.DataFrames loaded from csv files
        """
        csv_path = Path("~/.workintech/olist/data/csv").expanduser()

        file_paths = [p for p in csv_path.iterdir() if p.suffix == ".csv"]
        file_names = [p.name for p in file_paths]

        key_names = [
            n.replace(".csv", "").replace("_dataset", "").replace("olist_", "")
            for n in file_names
        ]

        data = {k: pd.read_csv(p) for k, p in zip(key_names, file_paths)}
        return data


    def ping(self):
        """
        You call ping I print pong.
        """
        print("pong")
