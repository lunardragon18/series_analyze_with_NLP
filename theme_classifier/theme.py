import pandas as pd
from transformers import pipeline
import huggingface_hub
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import load_dataset
import nltk
import numpy as np
from nltk import sent_tokenize
import torch

nltk.download('punkt')
nltk.download('punkt_tab')

class Theme:
    def __init__(self,theme_list):
        self.model = 'facebook/bart-large-mnli'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.theme_classifier = self.load_model()
        self.theme_list = theme_list

    def load_model(self):
        theme_classifier = pipeline(
            "zero-shot-classification",
            model=self.model,
            device=self.device
        )
        return theme_classifier

    def get_theme_scores(self, script):
        sentence = sent_tokenize(script)
        sentences = []
        batch_size = 20
        for i in range(0, len(sentence), batch_size):
            line = "".join(sentence[i:i + batch_size])
            sentences.append(line)
        model = self.theme_classifier
        outputs = model(
            sentences,
            self.theme_list,
            multi_label=True
        )
        themes = {}
        for output in outputs:
            for theme, score in zip(output["labels"], output["scores"]):
                if theme in themes:
                    themes[theme].append(score)
                else:
                    themes[theme] = [score]
        themes = {theme: np.mean(np.array(themes[theme])) for theme in themes}
        return themes

    def save_themes(self,dataset_path,save_path = None):
        if save_path is not None and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            return df
        df = load_dataset(dataset_path)
        themes = df["script"].apply(self.get_theme_scores)
        themes_df = pd.DataFrame(themes.tolist())
        df[themes_df.columns] = themes_df

        if save_path is not None:
            df.to_csv(save_path,index=False)
            return df





model_name = 'facebook/bart-large-mnli'

device = "cuda" if torch.cuda.is_available() else "cpu"
