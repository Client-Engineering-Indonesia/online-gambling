import warnings
warnings.filterwarnings('ignore')

import io
import os
import ast
from fastapi import FastAPI, HTTPException, Request
from fastapi.openapi.utils import get_openapi
import pandas as pd

# from app.helpers.helper import *
from app.helpers.wxwd_function import *

app = FastAPI(
    title='Sample-app FastAPI and Docker',
    version = '1.0.0',
)


@app.get("/")
async def root():
    return {"message": "Hello World with web identification..."}

@app.get("/ping")
async def ping():
    return "Hello, I am alive..."

# @app.post("/process_dict")
# async def process_dict(input_dict: dict):
#     # You can perform any processing on the input dictionary here
#     # For example, let's just return the received dictionary as is
#     return input_dict

@app.post("/content_gambling_category")
async def content_gambling_category(request: Request):

    try:
        user_input = await request.json()
        context = user_input['content']

        watson_context_category = WatsonQA()
        answer = await watson_context_category.gambling_category(context)
        return answer
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/content_search")
async def content_search(request: Request):

    try:
        user_input = await request.json()
        query = user_input['query']
        search_key = user_input['search_key']
        search_engine = user_input['search_engine']
        sort = user_input['sort']
        limit_results = user_input['limit_results']
        
        watson_context_category = WatsonQA()
        answer = await watson_context_category.advanced_search(query,search_key,search_engine,sort,limit_results)
        return answer
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/get_detailed_playid")
async def get_detailed_playid(request: Request):

    try:
        user_input = await request.json()
        query = user_input['query']
        lim_results = user_input['lim_results']

        watson_context_category = WatsonQA()
        answer = await watson_context_category.search_play(query, lim_results)
        return answer
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/get_review_context")
async def get_review_context(request: Request):

    try:
        user_input = await request.json()
        app_id = user_input['app_id']
        lim_reviews = user_input['lim_reviews']

        watson_context_category = WatsonQA()
        answer = await watson_context_category.review_play(app_id,lim_reviews)
        return answer
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/get_review")
async def get_review(request: Request):

    try:
        user_input = await request.json()
        app_id = user_input['app_id']
        lim_reviews = user_input['lim_reviews']

        watson_context_category = WatsonQA()
        answer = await watson_context_category.review_play_one(app_id,lim_reviews)
        return answer
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/get_review_multiple")
async def get_review_multiple(request: Request):

    try:
        user_input = await request.json()
        query = user_input['query']
        lim_results = user_input['lim_results']
        lim_reviews = user_input['lim_reviews']

        watson_context_category = WatsonQA()
        answer = await watson_context_category.review_play_multiple(query, lim_results, lim_reviews)
        return answer
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/reverse_image_search")
async def image_search(request: Request):

    try:
        user_input = await request.json()
        image_query = user_input['image_query']
        search_key = user_input['search_key']
        num_pages = user_input['num_pages']

        watson_context_category = WatsonQA()
        answer = await watson_context_category.reverse_image_search(image_query,search_key,num_pages)
        return answer
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Custom title",
        version="3.0.2",
        description="Here's a longer description of the custom **OpenAPI** schema",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    openapi_schema["servers"] = [{"url": "http://localhost:8000"}]  # Add your server URL
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi