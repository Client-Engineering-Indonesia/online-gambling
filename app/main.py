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