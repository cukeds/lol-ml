import pandas as pd

df_participants = pd.read_json("./scraped/participants_list.json")

print(df_participants.info())

