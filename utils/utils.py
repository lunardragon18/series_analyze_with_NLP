from glob import glob
import pandas as pd



def load_dataset(dataset_path):
  dataset = glob(dataset_path+'/*.ass')
  script = []
  episode_num = []
  for path in dataset:
    with open(path,'r',encoding='utf-8') as file:
      lines = file.readlines()
      lines = lines[27:]
      text = [line.split(",,")[-1] for line in lines]
    sentences = " ".join(text).replace("\\N"," ")
    script.append(sentences)
    episode_num.append(int(path.split('-')[-1].split('.')[0].strip()))
  df = pd.DataFrame({'episode_num':episode_num,'script':script})
  return df.sort_values(by='episode_num',ascending=True)