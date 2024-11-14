import data.data_loader
import pandas as pd

from data.data_loader import load_spotify_dataset

def main():
    # Step 1: Load and preprocess data
    print("Loading and preprocessing data...")
    df_spotify, target_column = load_spotify_dataset()



if __name__ == "__main__":
    main()