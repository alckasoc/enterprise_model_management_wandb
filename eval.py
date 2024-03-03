import os
import random
import wandb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # This will load all the environment variables from the .env file.
wandb_api_key = os.getenv('WANDB_API_KEY')
wandb.login(key=wandb_api_key)

alias = "production"

def main():
    # Load reference/baseline result (Gemma-2B-IT).
    run = wandb.init()
    artifact = run.use_artifact('vincenttu/enterprise_model_management_wandb/run-zijpzlbe-val_table_v1:v0', type='run_table')
    table = artifact.get("val_table_v1")
    ref_df = pd.DataFrame(data=table.data, columns=table.columns)
    run.finish()

    # Load new result (Gemma-7B-IT).
    run = wandb.init(project="enterprise_model_management_wandb")
    artifact = run.use_artifact(f'vincenttu/model-registry/gemma-2b:{alias}', type='model')
    producer_run_id = artifact.logged_by().id
    table_artifact = wandb.use_artifact(f"run-{producer_run_id}-val_table:v0")
    table = table_artifact.get("val_table")
    new_df = pd.DataFrame(data=table.data, columns=table.columns)
    run.finish()

    # Merge the DFs.
    merged_df = pd.merge(
        ref_df, 
        new_df, on="id", suffixes=["_ref", "_new"],
        how='inner'
    )[["id", "input_text_ref", "output_ref", "output_new", "target_ref"]]
    merged_df = merged_df.rename(columns={"input_text_ref": "input_text", "target_ref": "target"})

    # Compare the results (randomly).
    choices = ["output_ref", "output_new"]

    merged_df['choice'] = merged_df.apply(lambda _: random.choice(choices), axis=1)
    columns_to_include = ['id', 'input_text', 'output_ref', 'output_new', 'choice', 'target']
    out_df = merged_df[columns_to_include]
    table = wandb.Table(dataframe=out_df)

    # Log the table.
    run = wandb.init(project="enterprise_model_management_wandb", name="production_compare")
    run.log({"production_compare": table})
    run.finish()


if __name__ == "__main__":
    main()
    exit(0)