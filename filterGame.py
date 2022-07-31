import os
from fnmatch import fnmatch
import json
import numpy as np
import traceback

# root = "/content/content/SoccerNetVideo"
# patternAudio = "1*p.npy"
# patternVisual = "1*TF2.npy"
audioFeatureList = []
visualFeatureList = []
# for path, subdirs, files in os.walk(root):
#     for name in files:
#         if fnmatch(name, patternAudio):
#             print("Audio", os.path.join(path, name))
#             audioFeatureList.append((np.load(os.path.join(path, name)), os.path.join(path, name)))
#         if fnmatch(name, patternVisual):
#             print("Visual", os.path.join(path, name))            
#             visualFeatureList.append((np.load(os.path.join(path, name)), os.path.join(path, name)))
rootAudio = "/content/drive/MyDrive/Thesis_temp/soccernet-video"
rootVisual = "/content/TemporallyAwarePooling_Data/content/SoccerNet/content/TemporallyAwarePooling/SoccerNet_TemporallyAwarePooling"
from SoccerNet.Downloader import getListGames
print(getListGames('train'))

allGames = []

allGames.append(getListGames('train'))
allGames.append(getListGames('test'))
allGames.append(getListGames('valid'))

errorGames = [[],[],[]]
# len(errorGames)
def getShapeWithoutLoading(numpyFile):
  with open(numpyFile, 'rb') as f:
      major, minor = np.lib.format.read_magic(f)
      shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
  return shape

for gameSet, setName, i in zip(allGames, ["train", "test", "valid"], range(3)):
    print(f"Getting set: {setName}")
    for game in gameSet:
        print(f"Getting game {game}")
        for half in [1, 2]:
            try:
              audioFileName = os.path.join(rootAudio, game, f"{half}_224p_VGGish.npy") 
              audioFeaturesShape = getShapeWithoutLoading(audioFileName)
              print(f"LOADING {game} HALF {half}..., SHAPE : {audioFeaturesShape}")


            #   visualFileName = os.path.join(rootVisual, game, f"{half}_ResNET_TF2.npy") 
            #   visualFeaturesShape = getShapeWithoutLoading(visualFileName)

              # diff = audioFeaturesShape[1] - visualFeaturesShape[0]

              # print(f"LOADING {game} HALF {half}..., SHAPE DIFF: {diff}")
              if(audioFeaturesShape[2]!=512):
                print(f"ERROR (LENGTH): {game}")
                errorGames[i].append(game)


            except Exception:
              traceback.print_exc()
              print(f"ERROR (LOAD): {game}")
              errorGames[i].append(game)
              continue
filteredGames = []
for gset, egset, i in zip(allGames, errorGames, range(3)):
  filteredGames.append(list(set(gset)-set(egset)))

for gs, i, setname in zip(filteredGames, range(3), ["train", "test", "valid"]):
  np.save(os.path.join("/content/splits", f"{setname}_split.npy"), gs)
for game in filteredGames:
  print(len(game))