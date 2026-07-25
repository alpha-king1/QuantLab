from dotenv import load_dotenv
from fastapi import FastAPI
from Backend.Api.Routes import router
app = FastAPI()
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv('FRONTEND_URL'),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)