from fastapi import APIRouter
from starlette import status

from Backend.Api.AnalyseRequest import AnalysisRequest
from Backend.Core.Engine import Analyse
from Backend.Data.LoadData import LoadData

router = APIRouter()

@router.get("/")
def home():
    return {
        "message":"Welcome to QuantLab"
    }

@router.get("/trading_pairs")
def get_trading_pairs():
    loader = LoadData()
    return loader.get_pairs()


@router.get('/strategies')
def get_strategies():
    return \
    [
        {
            "id": "bullish_ob",
            "name": "Bullish Order Block"
        },
        {
            "id": "bearish_ob",
            "name": "Bearish Order Block"
        }
    ]


@router.get('/timeframe')
def get_timeframe():
    return [
        # "M1",
        # "M5",
        # "M15",
        "H1",
        "H4",
        "D"
    ]

@router.post('/analyse')
def analyse(request: AnalysisRequest):
    analysis = Analyse(pair=request.pair, granularity=request.timeframe, user_strategy=request.strategy, start_date=request.start_date, end_date=request.end_date, capital=request.capital)
    return analysis.run_engine()

