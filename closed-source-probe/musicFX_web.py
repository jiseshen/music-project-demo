import glob
import random
import requests
import os
from tqdm import tqdm
import pandas as pd
import dotenv
import json
import base64
import logging
from time import sleep

dotenv.load_dotenv(override=True)


class Music:
    musiclm_url = "https://content-aisandbox-pa.googleapis.com/v1:soundDemo?alt=json"

    def __init__(self):
        self.token = os.environ["GOOGLE_LAB_TOKEN"]

    def get_tracks(self, input, generationCount=1):
        generationCount = min(8, max(1, generationCount))

        payload = json.dumps({
            "generationCount": generationCount,
            "input": {
                "textInput": input
            },
            "soundLengthSeconds": 30  # this doesn't change anything
        })

        headers = {
            'Authorization': f'Bearer {self.token}'
        }
        try:
            response = requests.post(
                self.musiclm_url, headers=headers, data=payload)
        except requests.exceptions.ConnectionError:
            logging.error("Can't connect to the server.")
            # Bad Gateway
            return 502
        if response.status_code == 400:
            logging.error("Oops, can't generate audio for that.")
            # Bad Request
            return 400
        if response.status_code == 429:
            logging.error("Too many requests.")
            # Too Many Requests
            return 429
        tracks = []
        for sound in response.json()['sounds']:
            tracks.append(sound["data"])

        return tracks

    def b64toMP3(self, tracks_list, filename):
        with open(f"{filename}.mp3", "wb") as f:
            f.write(base64.b64decode(tracks_list[0]))
        # Successful Request
        return 200


if __name__ == "__main__":
    # music = Music()
    # df = pd.read_csv('datasets/MusicCaps/musiccaps-public.csv')
    # id2text = df.set_index('ytid')['caption'].to_dict()
    # generated_dir = f'musicFX/generated'
    # os.makedirs(generated_dir, exist_ok=True)
    # for id in tqdm(id2text):
    #     if os.path.exists(f'{generated_dir}/{id}.mp3'):
    #         continue
    #     tracks = music.get_tracks(id2text[id])
    #     music.b64toMP3(tracks, f'{generated_dir}/{id}')
    #     sleep(2)

    # music = Music()
    # df = pd.read_csv('handcraft_prompts.csv')
    # generated_dir = 'musicFX/prompts'
    # os.makedirs(generated_dir, exist_ok=True)
    # for id, prompt in enumerate(df['Prompts']):
    #     tracks = music.get_tracks(prompt)
    #     music.b64toMP3(tracks, f'{generated_dir}/{id}')
    #     sleep(2)

    music = Music()
    instruments = [
        "Piano",
        "Electric Piano",
        "Synthesizer",
        "Keyboard",
        "Organ",
        "Guitar",
        "Violin",
        "Cello",
        "Flute",
        "Saxophone",
        "Trumpet",
        "Trombone",
        "Marimba",
        "Xylophone",
        "Slap Bass",
        "Strings",
    ]

    genres = [
        "Pop",
        "Rock",
        "Metal",
        "Blues",
        "Jazz",
        "Funk",
        "Soul",
        "Hip Hop",
        "Rap",
        "R&B",
        "Reggae",
        "Country",
        "Folk",
        "Indie",
        "Electronic",
        "House",
        "Techno",
        "Classical",
        "Gospel",
        "Latin"
    ]

    moods = [
        "Uplifting",
        "Emotional",
        "Happy",
        "Inspiring",
        "Romantic",
        "Sad",
        "Melancholic",
        "Dark",
        "Upbeat",
        "Energetic",
        "Nostalgic",
        "Calm",
        "Hopeful",
        "Relaxing",
        "Dreamy",
        "Rebellious",
        "Confident",
        "Aggressive",
        "Positive",
        "Cool",
        "Angry",
        "Celebratory",
        "Bold",
        "Introspective",
        "Optimistic",
        "Sentimental",
        "Motivational",
        "Funky",
        "Danceable",
        "Fast-Paced",
        "Futuristic",
        "Joyful",
        "Soulful",
        "Epic",
        "Festive",
        "Exciting",
        "Powerful",
        "Passionate",
        "Playful",
        "Intense",
        "Sensual",
        "Adventurous",
        "Atmospheric",
        "Cheerful",
        "Chill",
        "Empowering",
        "Peaceful",
        "Reflective",
        "Pensive",
        "Lively",
        "Dramatic",
        "Gentle",
        "Groovy",
        "Spiritual"
    ]

    base_dir = 'musicFX/instruments'
    os.makedirs(base_dir, exist_ok=True)
    for instrument in tqdm(instruments):
        os.makedirs(f"{base_dir}/{instrument}", exist_ok=True)
        prompts = []
        labels = []
        if len(glob.glob(f"{base_dir}/{instrument}/*.mp3")) >= 10:
            continue
        for i in range(10):
            genre = random.choice(genres)
            mood = random.choice(moods)
            prompt = f"Compose a {genre} mainly featuring {instrument} with a beautiful {mood} melody."
            label = f"{instrument}_{genre}_{mood}"
            tracks = music.get_tracks(prompt)
            music.b64toMP3(tracks, f'{base_dir}/{instrument}/{label}')
            sleep(2)