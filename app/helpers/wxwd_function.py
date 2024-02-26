import json
import math
import pandas as pd
from ibm_watson.discovery_v2 import DiscoveryV2, QueryLargePassages
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
import os, re, ast
# from dotenv import load_dotenv

class WatsonQA:

    def __init__(self):
        # dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
        # load_dotenv(dotenv_path)

        # self.WX_API_KEY = os.getenv('WX_API_KEY')
        # self.WX_PROJECT_ID = os.getenv('WX_PROJECT_ID')
        # self.WX_URL = os.getenv('WX_URL')

        self.WX_API_KEY = os.environ['WX_API_KEY']
        self.WX_PROJECT_ID = os.environ['WX_PROJECT_ID']
        self.WX_URL = os.environ['WX_URL']

        # Initialize Watson XAI
        self.api_key_wx = self.WX_API_KEY
        self.ibm_cloud_url_wx = self.WX_URL
        self.project_id_wx = self.WX_PROJECT_ID
        self.creds_wx = {
            "url": self.ibm_cloud_url_wx,
            "apikey": self.api_key_wx
        }
    # ibm-mistralai/mixtral-8x7b-instruct-v01-q meta-llama/llama-2-13b-chat
    def send_to_watsonxai(self, prompts, model_name='ibm-mistralai/mixtral-8x7b-instruct-v01-q', decoding_method="greedy",
                          max_new_tokens=4095, min_new_tokens=1, temperature=0, repetition_penalty=1.0,
                          stop_sequences=["\n\n"]):
        assert not any(map(lambda prompt: len(prompt) < 1, prompts)), "make sure none of the prompts in the inputs prompts are empty"

        model_params = {
            GenParams.DECODING_METHOD: decoding_method,
            GenParams.MIN_NEW_TOKENS: min_new_tokens,
            GenParams.MAX_NEW_TOKENS: max_new_tokens,
            GenParams.RANDOM_SEED: 42,
            GenParams.TEMPERATURE: temperature,
            GenParams.REPETITION_PENALTY: repetition_penalty,
            GenParams.STOP_SEQUENCES: stop_sequences
        }

        api_key = self.WX_API_KEY
        ibm_cloud_url = self.WX_URL
        project_id = self.WX_PROJECT_ID

        creds = {
            "url": ibm_cloud_url,
            "apikey": api_key
        }

        model = Model(
            model_id=model_name,
            params=model_params,
            credentials=creds,
            project_id=project_id)

        for prompt in prompts:
            output = model.generate_text(prompt)

        return output

    async def gambling_category(self, context):
        # json_format = {
        #     "deskripsi": "rangkum informasi yang di temukan dalam 3 kalimat",
        #     "url": "sebutkan semua url yang ditemukan",
        #     "sentimen": "pandangan dari informasi yang ditemukan",
        # }

        # format = '{ "online_gambling_category": <nilai_persentase>, "value_negative_sentiment": <nilai_persentase>, "value_neutral_sentiment": <nilai_persentase>, "value_positive_sentiment": <nilai_persentase>, "penjelasan": "<penjelasan_dalam_Bahasa_Indonesia>", "url": "<url_yang_ditemukan>" }'

        format = '{ "online_gambling_category": <nilai_persentase>, "explanation": "<penjelasan_dalam_Bahasa_Indonesia>", "url": "<semua_url_yang_ditemukan>" }'
        
        prompt_stage = f"""Anda adalah asisten yang membantu, menghormati, dan jujur. Selalu jawab sebisa mungkin, sambil tetap aman. Jawaban Anda tidak boleh mengandung konten yang berbahaya, tidak etis, rasial, seksis, beracun, berbahaya, atau ilegal. Pastikan bahwa respons Anda tidak memihak dan bersifat positif.
        Article: {context}
        Tentukan kategori dari artikel yang diberikan dan berikan skornya menggunakan persentase untuk kategori judi online. Sediakan output dalam format JSON, dengan nilai yang mewakili persentase dari kategori sentimen. Pastikan total dari semua persentase adalah 1, mencerminkan kecenderungan keseluruhan dari artikel tersebut. Tampilkan dalam bentuk list untuk semua url apabila ditemukan dalam artikel yang diawali atau diakhiri dengan 'https atau wwww atau .com'. Buat penjelasan dalam Bahasa Indonesia dan formatkan sebagai berikut: {format}.

        Answer:"""
        output_stage = self.send_to_watsonxai(prompts=[prompt_stage], stop_sequences=[])
        output_stage = {"output": str(output_stage.strip()).replace('\n\n', ' ').replace('*', '<li>')}
        output_stage["output"] = re.sub(' +', ' ', output_stage["output"])
        output_stage["output"] = ast.literal_eval(output_stage['output'])
        
        return output_stage
    

        # prompt_stage = f"""Kamu adalah asisten yang membantu, menghormati, dan jujur. Selalu jawab sebisa mungkin, sambil tetap aman. Jawaban Anda tidak boleh mengandung konten yang berbahaya, tidak etis, rasial, seksis, beracun, berbahaya, atau ilegal. Pastikan bahwa respons Anda tidak memihak dan bersifat positif.
        # context: {context}
        # Tolong rangkum context dan informasi yang dapat ditemukan pada konteks di atas dalam 1 kalimat. Tampilkan url link dalam bentuk list apabila ditemukan dalam context. Jawaban harus mengikuti format json {json_format}"""
