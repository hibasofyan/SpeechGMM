import os
import numpy as np
import librosa
import pickle
from pydub import AudioSegment
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LanguageDetector:
    def __init__(self, models_dir="models_langues"):
        self.models_dir = models_dir
        self.models = {}
        self.load_models()
        
    def load_models(self):
        try:
            if not os.path.exists(self.models_dir):
                logging.error(f"Le dossier {self.models_dir} n'existe pas")
                return
                
            for filename in os.listdir(self.models_dir):
                if filename.endswith('.pkl'):
                    language = filename.replace('.pkl', '')
                    model_path = os.path.join(self.models_dir, filename)
                    try:
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                            if hasattr(model, 'score'):  
                                self.models[language] = model
                                logging.info(f"Modèle chargé pour la langue : {language}")
                            else:
                                logging.error(f"Le modèle pour {language} n'est pas un modèle GMM valide")
                    except Exception as e:
                        logging.error(f"Erreur lors du chargement du modèle {language}: {str(e)}")
                        
            if not self.models:
                logging.warning("Aucun modèle n'a pu être chargé")
                
        except Exception as e:
            logging.error(f"Erreur lors du chargement des modèles : {str(e)}")
            
    def preprocess_audio(self, audio_path, max_duration=5):
    
        try:
            y, sr = librosa.load(audio_path, sr=44100)
            
            if len(y) > max_duration * sr:
                y = y[:int(max_duration * sr)]
            
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc = mfcc.T
            
            return mfcc
            
        except Exception as e:
            logging.error(f"Erreur lors du prétraitement de l'audio : {str(e)}")
            return None
            
    def detect_language(self, audio_path):
        
        try:
            mfcc = self.preprocess_audio(audio_path)
            if mfcc is None:
                return None
                
            scores = {}
            for language, model in self.models.items():
                score = model.score(mfcc)
                scores[language] = score
                
            detected_language = max(scores.items(), key=lambda x: x[1])[0]
            logging.info(f"Langue détectée par GMM : {detected_language}")
            
            return detected_language
            
        except Exception as e:
            logging.error(f"Erreur lors de la détection de langue : {str(e)}")
            return None 