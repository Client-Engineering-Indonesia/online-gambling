import json
import math
import pandas as pd
from ibm_watson.discovery_v2 import DiscoveryV2, QueryLargePassages
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
import os, re, ast
import json, requests
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

    async def advanced_search(self, query, search_key, search_engine, sort, limit_results):

        url = "https://www.googleapis.com/customsearch/v1"

        params = {
            "q": query,
            "key": search_key,
            "cx": search_engine,
            # "dateRestrict": "d365",  # Search for the past 7 days
            "sort": "date" if sort == "date" else None,  # default is relevance
            "start": 1,  # Incrementing start value
            "num": 10,  # Number of results per page
        }
            
        start_value = 1
        collected_data = []

        while True:
            response = requests.get(url, params=params)
            response.raise_for_status() 
            search_results = response.json()

            if 'items' not in search_results:
                break
            
            for item in search_results['items']:
                title = item['title']
                link = item['link']
                snippet = item['snippet']
                # image = None
                # if 'pagemap' in item and 'cse_image' in item['pagemap']:
                #     image = item['pagemap']['cse_image'][0]['src']  # Assuming there is only one image

                item_data = {
                    'title': title,
                    'link': link,
                    'snippet': snippet,
                    # 'image': image
                }
                collected_data.append(item_data)
            
            start_value += 10

            if start_value >= limit_results:
                break

        json_data = json.dumps(collected_data, indent=0).replace('\n', '')

        return json_data