from tensorflow.keras.models import load_model
import ml
import pandas as pd
import numpy as np

matches = pd.read_csv("./scraped/grandmaster/matches")

i = 0
for match in matches["match_id"]:

    df_match = pd.read_json("./scraped/grandmaster/games/" + str(match) + ".json")

    if i == 0:
        test_data_players = pd.DataFrame(df_match["info"]["participants"])
        test_data_players = test_data_players.drop(["championId", "championName", "championTransform", "eligibleForProgression"], axis=1)
        test_data_players = test_data_players.drop(["item0", "item1", "item2", "item3", "item4", "item5", "item6", "neutralMinionsKilled", "participantId"], axis=1)
        test_data_players = test_data_players.drop(["perks", "profileIcon", "puuid", "riotIdName", "riotIdTagline", "role", "summoner1Id", "summoner2Id"], axis=1)
        test_data_players = test_data_players.drop(["summonerId", "summonerLevel", "summonerName", "teamId", "unrealKills", "lane"], axis=1)
        test_data_players = test_data_players.drop(["challenges"], axis=1)

    else:
        df_player = pd.DataFrame(df_match["info"]["participants"])
        df_player = df_player.drop(["championId", "championName", "championTransform", "eligibleForProgression"], axis=1)
        df_player = df_player.drop(["item0", "item1", "item2", "item3", "item4", "item5", "item6", "neutralMinionsKilled", "participantId"], axis=1)
        df_player = df_player.drop(["perks", "profileIcon", "puuid", "riotIdName", "riotIdTagline", "role", "summoner1Id", "summoner2Id"], axis=1)
        df_player = df_player.drop(["summonerId", "summonerLevel", "summonerName", "teamId",  "unrealKills", "lane"], axis=1)
        if "challenges" in df_player.columns:
            df_player = df_player.drop(["challenges"], axis=1)
        test_data_players = pd.concat([test_data_players, df_player])

    i+=1
    if i == 1000:
        break


test_data_players = test_data_players.reset_index(drop=True)
test_data_players = test_data_players[test_data_players["individualPosition"] != "INVALID"].drop("individualPosition", axis=1).reset_index(drop=True)
test_data_players = test_data_players[test_data_players["teamPosition"] != ""].reset_index(drop=True)
test_data_players = test_data_players.replace({True: 1, False: 0})
test_data_players = test_data_players.replace({"NONE": None})
test_data_players = test_data_players.dropna()
x = test_data_players.drop("teamPosition", axis=1).to_numpy(dtype="int32")
y = test_data_players["teamPosition"].replace({"TOP":0, "JUNGLE":1, "MIDDLE":2, "BOTTOM":3, "UTILITY":4}).to_numpy(dtype="int32")


# load the model from file
model = load_model('model.h5')
# make a prediction
yhat = model.predict(x)

pos = ["TOP", "JG", "MID", "ADC", "SUPP"]
predictions = [pos[np.argmax(data)] for data in yhat]
real = [pos[data] for data in y]

accuracy = [1 if predictions[i] == real[i] else 0 for i in range(len(predictions))]
print('Predicted:', predictions)
print('Real:', real)
print('Got it?:', accuracy)
print(accuracy.count(1) / len(accuracy))