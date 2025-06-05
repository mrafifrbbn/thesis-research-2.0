import pandas as pd

base_path = "/Users/mrafifrbbn/Documents/thesis/thesis-research-2.0/experiments/experiment_036_mock_malmquist_prior/"

def main():
    for mock_id in [1, 2, 3, 4, 5]:

        # Load mock with 13.65 limit
        filepath = base_path + f"logdists/mock_{mock_id}/mock_13.65.csv"
        df = pd.read_csv(filepath)

        # Load other mock
        for mag_lim in ["13.15", "12.65", "12.15"]:
            filepath = base_path + f"logdists/mock_{mock_id}/mock_{mag_lim}.csv"
            df_ = pd.read_csv(filepath)[["#mockgal_ID", f"logdist_{mag_lim}", f"logdist_{mag_lim}_err"]]

            # Merge with the 13.65
            df = df.merge(df_, on="#mockgal_ID", how="left")

        # Save
        filepath = base_path + f"logdists/mock_{mock_id}/mock_combined.csv"
        df.to_csv(filepath, index=False)

    print("Combining mocks successful!")


if __name__ == "__main__":
    main()