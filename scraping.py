import warnings

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ml
import os
import shap




if os.path.isfile("./model.png"):
    os.remove("model.png")


matches = pd.read_csv("./scraped/challenger/matches")

i = 0
for match in matches["match_id"]:

    df_match = pd.read_json("./scraped/challenger/games/" + str(match) + ".json")

    if i == 0:
        train_data_players = pd.DataFrame(df_match["info"]["participants"])
        train_data_players = train_data_players.drop(["championId", "championName", "championTransform", "eligibleForProgression"], axis=1)
        train_data_players = train_data_players.drop(["item0", "item1", "item2", "item3", "item4", "item5", "item6", "neutralMinionsKilled", "participantId"], axis=1)
        train_data_players = train_data_players.drop(["perks", "profileIcon", "puuid", "riotIdName", "riotIdTagline", "role", "summoner1Id", "summoner2Id"], axis=1)
        train_data_players = train_data_players.drop(["summonerId", "summonerLevel", "summonerName", "teamId", "unrealKills", "lane"], axis=1)
        train_data_players = train_data_players.drop(["challenges"], axis=1)
        i=1
    else:
        df_player = pd.DataFrame(df_match["info"]["participants"])
        df_player = df_player.drop(["championId", "championName", "championTransform", "eligibleForProgression"], axis=1)
        df_player = df_player.drop(["item0", "item1", "item2", "item3", "item4", "item5", "item6", "neutralMinionsKilled", "participantId"], axis=1)
        df_player = df_player.drop(["perks", "profileIcon", "puuid", "riotIdName", "riotIdTagline", "role", "summoner1Id", "summoner2Id"], axis=1)
        df_player = df_player.drop(["summonerId", "summonerLevel", "summonerName", "teamId",  "unrealKills", "lane"], axis=1)
        if "challenges" in df_player.columns:
            df_player = df_player.drop(["challenges"], axis=1)
        train_data_players = pd.concat([train_data_players, df_player])


train_data_players = train_data_players.reset_index(drop=True)
train_data_players = train_data_players[train_data_players["individualPosition"] != "INVALID"].drop("individualPosition", axis=1).reset_index(drop=True)
train_data_players = train_data_players[train_data_players["teamPosition"] != ""].reset_index(drop=True)
train_data_players = train_data_players.replace({True: 1, False: 0})
train_data_players = train_data_players.replace({"NONE": None})
train_data_players = train_data_players.dropna()
x = train_data_players.drop("teamPosition", axis=1).to_numpy(dtype="int32")[:-20]
y = train_data_players["teamPosition"].replace({"TOP":0, "JUNGLE":1, "MIDDLE":2, "BOTTOM":3, "UTILITY":4}).to_numpy(dtype="int32")

# DeepNN
### layer input
n_features = 91
inputs = ml.layers.Input(name="input", shape=(n_features,))
### hidden layer 1
h1 = ml.layers.Dense(name="h1", units=128, activation='sigmoid')(inputs)
h1 = ml.layers.Dropout(name="drop1", rate=0.2)(h1)
### hidden layer 2
h2 = ml.layers.Dense(name="h2", units=128, activation='sigmoid')(h1)
h2 = ml.layers.Dropout(name="drop2", rate=0.2)(h2)
### layer output
outputs = ml.layers.Dense(name="output", units=5, activation='softmax')(h2)
model = ml.models.Model(inputs=inputs, outputs=outputs, name="DeepNN")

ml.utils.plot_model(model, to_file="model.png", show_shapes=True, show_layer_names = True)

# compile the neural network
model.compile(optimizer="adam", loss='categorical_crossentropy',
              metrics=['accuracy'])

## Train

y = ml.utils.to_categorical(y)
training = model.fit(x=x, y=y, batch_size=32, epochs=100,
                     shuffle=True, verbose=0, validation_split=0.3)

# plot
metrics = [k for k in training.history.keys() if ("loss" not in k)
           and ("val" not in k)]
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15, 3))

## training
ax[0].set(title="Training")
ax11 = ax[0].twinx()
ax[0].plot(training.history['loss'], color='black')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss', color='black')
for metric in metrics:
    ax11.plot(training.history[metric], label=metric)
    ax11.set_ylabel("Score", color='steelblue')
ax11.legend()

## validation
ax[1].set(title="Validation")
ax22 = ax[1].twinx()
ax[1].plot(training.history['val_loss'], color='black')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss', color='black')
for metric in metrics:
    ax22.plot(training.history['val_' + metric], label=metric)
    ax22.set_ylabel("Score", color="steelblue")
plt.show()

model.save('model.h5')