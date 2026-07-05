from fastapi import FastAPI

from Backend.Api.Routes import router

app = FastAPI()

app.include_router(router)