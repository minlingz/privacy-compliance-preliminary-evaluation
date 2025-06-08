import mysql.connector
import traceback
import pandas as pd
from dataclasses import dataclass
from trees import df_to_tree
import json
import numpy as np
from typing import List
from binning import get_bin_edges, assign_bins
import os
from scipy.stats import entropy

config = {
    "username": "onboarding",
    "host": "demodata.subsalt.io",
    "password": "",
    "database": "mids",
}

N_BINS = 25  # Assuming all cols use the 25 bins method


@dataclass
class Table:
    name: str
    indirect_ids: list
    columns_used: list
    datetime_columns: list
    numeric_indirect_ids: list


table_cache = {}  # Cache tables so we dont have to keep hitting the database


def get_table(config: dict, table_name: str, row_count: int) -> pd.DataFrame:
    """
    Get a single table by name
    """
    if table_name in table_cache:
        return table_cache[table_name].copy()
    try:
        connection = mysql.connector.connect(**config)

        cursor = connection.cursor()

        # Get the list of tables
        cap_row_count = min(row_count, 400000)
        cursor.execute(f"SELECT * FROM {table_name} ORDER BY RAND() LIMIT 5000")

        data = cursor.fetchall()
        df = pd.DataFrame(data, columns=[i[0] for i in cursor.description])
        df.columns = df.columns.str.strip()

        table_cache[table_name] = df.copy()

        return df

    except Exception as err:
        print(traceback.format_exc())
        return pd.DataFrame()


tree_stat_cache = {}  # Cache tree stats so we dont have to keep recalculating


def get_tree_stats(df: pd.DataFrame) -> dict:
    """
    Generate some statistics about the table using an EquivalenceClassTree.

    Default to building the tree with columns in the order they are in the dataframe
    """
    if tuple(df.columns.tolist()) in tree_stat_cache:
        return tree_stat_cache[tuple(df.columns.tolist())]

    layers = df.columns.tolist()
    tree = df_to_tree(df, layers)
    print(
        f"Created tree with {len(tree.get_nodes_for_layer(layers[-1]))} nodes on the bottom layer"
    )

    # Get bottom layer and get upstream sibling counts
    bottom_layer_nodes = tree.get_nodes_for_layer(layers[-1])
    cts = [node.upstream_sibling_counts for node in bottom_layer_nodes]

    # The upstream sibling counts act as a sort of uniqueness metric for the data
    # If a node has few sibling counts, it is more unique. If its parent has few sibling counts, it is more unique, etc.

    sibling_ct_df = pd.DataFrame(
        cts, columns=layers[::-1]
    )  # Sibling counts go from bottom to top, so reverse the layers to match
    sibling_ct_df["equivalence_class_size"] = [node.size for node in bottom_layer_nodes]

    # Compute the ratio of uniqueness on the bottom layer to one layer up
    # A ratio of 0 means the bottom layer is as unique as the layer above it. A ratio < 0 means it is more unique, > 0 means less unique
    # A more unique layer has fewer siblings so the ratio will be negative as the upper layer would be larger
    try:
        sibling_ct_df["uniqueness_ratio"] = (
            sibling_ct_df[layers[-1]] - sibling_ct_df[layers[-2]]
        ) / sibling_ct_df[layers[-1]]
    except IndexError:
        sibling_ct_df["uniqueness_ratio"] = -1

    out = {
        "num_equivalence_classes": len(bottom_layer_nodes),
        "avg_equivalence_class_size": sibling_ct_df["equivalence_class_size"].mean(),
        "median_equivalence_class_size": sibling_ct_df[
            "equivalence_class_size"
        ].median(),
        "std_equivalence_class_size": sibling_ct_df["equivalence_class_size"].std(),
        "num_small_equivalence_classes": sum(
            sibling_ct_df["equivalence_class_size"] < 5
        ),
        "avg_uniqueness_ratio": sibling_ct_df["uniqueness_ratio"].mean(),
        "uniqueness_ratio_small_classes": sibling_ct_df[
            sibling_ct_df["equivalence_class_size"] < 5
        ]["uniqueness_ratio"].mean(),
        # Compute entropy of the equivalence class sizes
        "ec_entropy": entropy(
            sibling_ct_df["equivalence_class_size"].value_counts(normalize=True)
        ),
        "num_rows_in_small_equivalence_classes": sibling_ct_df.loc[
            sibling_ct_df["equivalence_class_size"] < 5, "equivalence_class_size"
        ].sum(),
    }
    tree_stat_cache[tuple(df.columns.tolist())] = out
    return out


def get_api_data() -> List[Table]:
    """
    Get data from the API. The API is in the UI behind auth, so the easiest way to do this is to
    just hit the API in the UI and copy paste it into a json file that gets loaded here

    This function then takes that API data and puts it into a dict with the key being the columns. We dont
    have table name so we can just use a sorted list of columns as the key to join to the other data
    """
    filepath = "0320.json"
    with open(filepath, "r") as f:
        data = json.load(f)

    runs = []
    for table in data:
        table_name = table["metadata"].get("table_names", "UNKNOWN_TABLE")
        num_indirect_ids = sum([i["indirect_identifier"] for i in table["schema"]])
        num_iids_categorical = sum(
            [
                i["indirect_identifier"]
                and i["synthesize_as"] in ["categorical", "binary"]
                for i in table["schema"]
            ]
        )
        unique_values = [
            i["unique_values"]
            for i in table["schema"]
            if i["unique_values"] is not None
        ]
        avg_unique_values = np.mean(unique_values) if unique_values else -1
        max_unique_values = max(unique_values) if unique_values else -1
        std_unique_values = np.std(unique_values) if unique_values else -1

        num_datetime = sum([i["type"] == "datetime" for i in table["schema"]])
        num_string = sum([i["type"] == "string" for i in table["schema"]])

        indirect_id_unique_values = [
            i["unique_values"] for i in table["schema"] if i["indirect_identifier"]
        ]
        avg_indirect_id_unique_values = (
            np.mean(indirect_id_unique_values) if indirect_id_unique_values else -1
        )

        # add the average number of unique values for indirect identifiers that categorical
        iids_categorical_unique_values = [
            i["unique_values"]
            for i in table["schema"]
            if (
                i["indirect_identifier"]
                & (i["synthesize_as"] in ["categorical", "binary"])
            )
        ]
        avg_iids_categorical_unique_values = (
            np.mean(iids_categorical_unique_values)
            if iids_categorical_unique_values
            else -1
        )

        max_indirect_id_unique_values = (
            max(indirect_id_unique_values) if indirect_id_unique_values else -1
        )
        std_indirect_id_unique_values = (
            np.std(indirect_id_unique_values) if indirect_id_unique_values else -1
        )

        privacy_scores = {
            f"PRIVACY_{i['name']}": i["score"] for i in table["privacy"]
        }  # Anything with the name PRIVACY_ is a target not an input for models
        privacy_passes = {
            f"PRIVACY_{i['name']}_PASS": i["passed"] for i in table["privacy"]
        }  # Anything with the name PRIVACY_ is a target not an input for models

        this_run = {
            "table_name": table_name,
            "pct_indirect_ids": num_indirect_ids / len(table["schema"]),
            "avg_unique_values": avg_unique_values,
            "max_unique_values": max_unique_values,
            "std_unique_values": std_unique_values,
            "pct_datetime": num_datetime / len(table["schema"]),
            "pct_string": num_string / len(table["schema"]),
            "avg_indirect_id_unique_values": avg_indirect_id_unique_values,
            "max_indirect_id_unique_values": max_indirect_id_unique_values,
            "std_indirect_id_unique_values": std_indirect_id_unique_values,
            "row_count": table["metadata"]["row_count"],
            "cap_row_count": np.minimum(table["metadata"]["row_count"], 400000),
            "r_iids_categorical": num_iids_categorical / len(table["schema"]),
            "avg_iids_categorical_unique_values": avg_iids_categorical_unique_values,
        }

        # Add privacy scores
        this_run.update(privacy_scores)
        this_run.update(privacy_passes)

        # Add generator config
        this_run.update(table["genconfig"])

        runs.append(
            (
                Table(
                    name=table_name,
                    indirect_ids=[
                        i["name"] for i in table["schema"] if i["indirect_identifier"]
                    ],
                    columns_used=[
                        i["name"]
                        for i in table["schema"]
                        if i["name"] != "__subsalt_idx"
                    ],
                    datetime_columns=[
                        i["name"] for i in table["schema"] if i["type"] == "datetime"
                    ],
                    numeric_indirect_ids=[
                        i["name"]
                        for i in table["schema"]
                        if i["indirect_identifier"]
                        and i["synthesize_as"] not in ("categorical", "binary")
                    ],
                ),
                this_run,
            )
        )

    return runs


def map_table_columns(table: Table, df: pd.DataFrame) -> pd.DataFrame:
    """
    There are some minor inconsistencies between the API data and the actual data in the database. This function just hard codes manual fixes.
    """
    if (
        table.name == "StudentPerformanceFactors"
        and "student_performance_StudentPerformanceFactors_" in table.columns_used[0]
    ):
        df.columns = [
            "student_performance_StudentPerformanceFactors_" + col for col in df.columns
        ]

    elif table.name == "healthcare_dataset":
        if "healthcare_sample_healthcare_dataset_" in table.columns_used[0]:
            df.columns = [
                "healthcare_sample_healthcare_dataset_" + col for col in df.columns
            ]
        if "test_db_healthcare_dataset_" in table.columns_used[0]:
            df.columns = ["test_db_healthcare_dataset_" + col for col in df.columns]
        if "sample_data_healthcare_dataset_" in table.columns_used[0]:
            df.columns = ["sample_data_healthcare_dataset_" + col for col in df.columns]
    elif table.name == "credit_card_sample":
        if "credit_swipes_credit_card_sample_" in table.columns_used[0]:
            df.columns = [
                "credit_swipes_credit_card_sample_" + col for col in df.columns
            ]
        if "sample_data_credit_card_sample_" in table.columns_used[0]:
            df.columns = ["sample_data_credit_card_sample_" + col for col in df.columns]
    elif table.name == "adult":
        if "mids_adult_" in table.columns_used[0]:
            df.columns = ["mids_adult_" + col for col in df.columns]
    elif table.name == "hr_demo_data":
        if "mids_hr_demo_data_" in table.columns_used[0]:
            df.columns = ["mids_hr_demo_data_" + col for col in df.columns]
    elif table.name == "indicators":
        if "mids_indicators_" in table.columns_used[0]:
            df.columns = ["mids_indicators_" + col for col in df.columns]
    elif table.name == "border_crossing":
        if "spring_term_border_crossing_" in table.columns_used[0]:
            df.columns = ["spring_term_border_crossing_" + col for col in df.columns]
    elif table.name == "insurance_complaints":
        if "spring_term_insurance_complaints_" in table.columns_used[0]:
            df.columns = [
                "spring_term_insurance_complaints_" + col for col in df.columns
            ]
    elif table.name == "dog_bites_nyc":
        if "spring_term_dog_bites_nyc_" in table.columns_used[0]:
            df.columns = ["spring_term_dog_bites_nyc_" + col for col in df.columns]
    elif table.name == "parking_tickets_dc":
        if "spring_term_parking_tickets_dc_" in table.columns_used[0]:
            df.columns = ["spring_term_parking_tickets_dc_" + col for col in df.columns]
    elif table.name == "real_estate_2001_2022":
        if "spring_term_real_estate_2001_2022_" in table.columns_used[0]:
            df.columns = [
                "spring_term_real_estate_2001_2022_" + col for col in df.columns
            ]
    elif table.name == "loans_development_credit":
        if "spring_term_loans_development_credit_" in table.columns_used[0]:
            df.columns = [
                "spring_term_loans_development_credit_" + col for col in df.columns
            ]
    elif table.name == "recidivism_ny":
        if "spring_term_recidivism_ny_" in table.columns_used[0]:
            df.columns = ["spring_term_recidivism_ny_" + col for col in df.columns]
    elif table.name == "vehicle":
        if "spring_term_vehicle_" in table.columns_used[0]:
            df.columns = ["spring_term_vehicle_" + col for col in df.columns]
    elif table.name == "school_quality_reports":
        if "spring_term_school_quality_reports_" in table.columns_used[0]:
            df.columns = [
                "spring_term_school_quality_reports_" + col for col in df.columns
            ]
    elif table.name == "crime_la_service_calls":
        if "spring_term_crime_la_service_calls_" in table.columns_used[0]:
            df.columns = [
                "spring_term_crime_la_service_calls_" + col for col in df.columns
            ]
    elif table.name == "crime_baton_rouge_incidents":
        if "spring_term_crime_baton_rouge_incidents_" in table.columns_used[0]:
            df.columns = [
                "spring_term_crime_baton_rouge_incidents_" + col for col in df.columns
            ]
    elif table.name == "crime_nyc_arrests":
        if "spring_term_crime_nyc_arrests_" in table.columns_used[0]:
            df.columns = ["spring_term_crime_nyc_arrests_" + col for col in df.columns]
    elif table.name == "nursery":
        if "data_try_nursery_" in table.columns_used[0]:
            df.columns = ["data_try_nursery_" + col for col in df.columns]

    return df


def create_tables_df(input_from_api: List[Table]) -> pd.DataFrame:
    """
    Combines data pulled from the API with the data calc'd directly from the database to create a training dataframe
    """
    rows = []
    progress = 0
    for table, api_data in input_from_api:
        # only run the table if we know what it is the predefined list
        if table.name not in [
            "patdemo",
            "mimic",
            "census_10m",
            "census_5m",
        ]:  # skip the tables we dont know

            print(f"Processing run of table: {table.name}")
            # Get the table data
            df = get_table(config, table.name, api_data["row_count"])
            if df.empty:
                print(f"Table {table.name} is empty. Skipping.")
                continue
            print(
                f"Going to map columns for table: {table.name}. Columns in table: {table.columns_used}. Columns in df: {df.columns}"
            )
            df = map_table_columns(table, df)

            df = df[table.columns_used]  # Only use the columns that were in the run

            for column in table.datetime_columns:
                print(f"Converting column: {column} to unix timestamp")
                # Convert to datetime then to int unix timestamp
                df[column] = pd.to_datetime(df[column])
                df[column] = df[column].astype(int) // 10**9

            for column in table.numeric_indirect_ids:
                print(f"Binning column: {column}")
                bin_edges = get_bin_edges(df[column], N_BINS)
                df[column] = assign_bins(df[column].values, bin_edges)

            # Get tree stats but only use the indirect IDs as those define the equivalence classes
            tree_stats = get_tree_stats(df[table.indirect_ids])
            this_row = api_data.copy()
            this_row.update(tree_stats)
            rows.append(this_row)
        else:
            print(f"API entry didnt have a table listed, its probably old. Skipping.")

        progress += 1
        print(f"Progress: {progress}/{len(input_from_api)}")

    pd.DataFrame(rows).to_csv("all_data_fix.csv", index=False)


if __name__ == "__main__":
    api_data = get_api_data()

    training_df = create_tables_df(api_data)
