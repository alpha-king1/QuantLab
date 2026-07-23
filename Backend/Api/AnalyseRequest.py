from pydantic import BaseModel

class AnalysisRequest(BaseModel):

    strategy: str

    pair: str

    timeframe: str

    start_date: str

    end_date: str

    capital: float