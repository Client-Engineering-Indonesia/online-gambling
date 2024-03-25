from ibm_watson.discovery_v2 import DiscoveryV2, QueryLargePassages
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from google_play_scraper import search, Sort, reviews_all
import serpapi
import os, re, ast
import json, requests
import pandas as pd
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

    ######## category content webpage ########
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

    ######## search engine search ########
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
    
    ######## play appid search ########
    async def search_play(self, query, lim_results):
        """
        Get details of apps from a Google Play Store search query.

        Args:
            query (str): The search query.
            lang (str): Language code (defaults to 'id').
            country (str): Country code (defaults to 'id').
            num_results (int): Number of results to retrieve (defaults to 5).

        Returns:
            list: A list of dictionaries containing app details.
        """
        results = search(query=query, lang='id', country='id')

        app_details_list = []
        for result in results[:lim_results]:
            app_details = {
                "appId": result["appId"],
                "title": result["title"],
                "screenshots": result["screenshots"],
                "descriptionHTML": result["descriptionHTML"]
            }
            app_details_list.append(app_details)

        return app_details_list

    ######## get review from play appid ########
    async def review_play(self, app_id, lim_reviews):

        scrapreview = reviews_all(
            app_id,
            lang='id',
            country='id',
            sort=Sort.MOST_RELEVANT,
            count=1000,
            filter_score_with=None
        )

        app_reviews_df = pd.DataFrame(scrapreview)

        def labeling(row):
            if row <= 3:
                return "negatif"
            else:
                return "positif"

        # Select relevant columns
        data = app_reviews_df[["userName", "content", "score", "at"]].copy()  # Use .copy() to avoid the warning

        # Apply labeling function
        data["label"] = data["score"].apply(labeling)
        data = data.dropna(subset=["content"])

        data['word_count'] = data['content'].apply(lambda x: len(str(x).split()))
        data = data.loc[data['word_count'] > 5]  # Use .loc[] to avoid the warning

        content_list = data["content"].to_list()
        content = content_list[:lim_reviews]

        return content
    
    ######## category review play from appid ########
    def gambling_play_category(self, context):

        format = '{ "description": <deskripsi_game>, "sentiment": "<sentimen_review>", "indication": "<indikasi_apabila_ada_kata_judi>" }'

        # template = "<[INST] «SYS»\n"\
        # "while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.\n"\
        # "Please ensure that your responses are socially unbiased and positive in nature.\n"\
        # "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.\n"\
        # "If you don't know the answer to a question, please don't share false information.\n"\
        # "«/SYS»\n"\
        # "You are an HR Consultant that need to review an Organizational Structure's Job description from several departments. In the same division, you will have several task.\n"\
        # "For this process I want you to do a comprehensive understanding from this Job Description!\n"\
        # f"{input_data}"\
        # "Here are your task:\n"\
        # "1. List down all redundant job description from one department to another and make sure you write it very comprehensive!\n"\
        # "[/INST]\n"\
        # "Write the redundant job description per department n"\
        # "Output:"

        prompt_stage = f"""<[INST] «SYS» Anda adalah asisten yang membantu, menghormati, dan jujur. Selalu jawab sebisa mungkin, sambil tetap aman. Jawaban Anda tidak boleh mengandung konten yang berbahaya, tidak etis, rasial, seksis, beracun, berbahaya, atau ilegal. Pastikan bahwa respons Anda tidak memihak dan bersifat positif. «SYS»
        review: {context}
        Tolong deskripsikan konten review dari game yang ditemukan, bagaimana sentimen dari konten review tersebut, dan jelaskan apakah ada indikasi dari review tersebut apakah ada kata-kata yang menandakan adanya unsur perjudian dalam permainan.
        [/INST] Buat penjelasan ke dalam Bahasa Indonesia dan formatkan sebagai berikut: {format}
        Answer:"""
        output_stage = self.send_to_watsonxai(prompts=[prompt_stage], stop_sequences=[])
        output_stage = {"output": str(output_stage.strip()).replace('\n\n', ' ').replace('*', '<li>')}
        output_stage["output"] = re.sub(' +', ' ', output_stage["output"])
        # output_stage["output"] = ast.literal_eval(output_stage['output'])
        
        return output_stage
    
    ######## get 1 content review play from appid ########
    async def review_play_one(self, app_id, lim_reviews):
        
        content = self.review_play(app_id, lim_reviews)
        result = self.gambling_play_category(content)

        return result

    ######## get multiple content review play from appid ########
    async def review_play_multiple(self, query, lim_results, lim_reviews):

        multiple_search = self.search_play(query, lim_results)

        results_list = []

        for search_item in multiple_search:
            app_id = search_item["appId"]
            context = self.review_play(app_id, lim_reviews)
            result = self.gambling_play_category(context)
            results_list.append({app_id: result})

        return results_list
    
    ######## search reverse image ########
    async def reverse_image_search(self, image_url, search_key, num_pages):
        """
        Performs iterative reverse image search, retrieving results from potentially multiple pages.

        Args:
            image_url: URL of the image to search for.
            serpapi_key: Your SerpApi key.
            num_pages: Maximum number of pages to search (including the first page).

        Returns:
            A list of image results from all searched pages.
        """

        all_results = []
        page_num = 1

        while page_num <= num_pages:
            params = {
                "engine": "google_reverse_image",
                "lang": "id",
                "country":'id',
                "image_url": image_url,
                "api_key": search_key,
                "start": (page_num - 1) * 10,  # Adjust based on search engine's page size
                "output": "json"
            }

            search = serpapi.search(params)
            results = search.get("image_results", [])

            # Check if results are empty or there's an indication of no more pages
            if not results:
                break

            all_results.extend(results)
            page_num += 1

        ##if all_results not none
        extracted_results = []
        if all_results:
            for result in all_results:
                extracted_result = {
                    'position': result.get('position', None),
                    'title': result.get('title', None),
                    'link': result.get('link', None),
                    'snippet': result.get('snippet', None)
                }
                extracted_results.append(extracted_result)
        else:
            extracted_results = extracted_results

        return extracted_results