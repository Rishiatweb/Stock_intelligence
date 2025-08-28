from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta
import pymongo
import traceback
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta, date
import requests
from transformers import pipeline
import numpy as np
import talib  # You'll need to install this: pip install ta-lib

# --- (Keep your existing initializations for sentiment, db, etc.) ---
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["stock_db"]
collection = db["stock_prices"]
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Authentication setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
SECRET_KEY = "your-secret-key-here"  # In production, use a secure secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class User(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# Create a new users collection in MongoDB
users_collection = db["users"]

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(username: str):
    user_dict = users_collection.find_one({"username": username})
    if user_dict:
        return UserInDB(**user_dict)

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.JWTError:
        raise credentials_exception
    user = get_user(username)
    if user is None:
        raise credentials_exception
    return user
companies = [
    {"symbol": "AAPL", "name": "Apple Inc."}, {"symbol": "MSFT", "name": "Microsoft Corp."},
    {"symbol": "GOOGL", "name": "Alphabet Inc. (Google)"}, {"symbol": "AMZN", "name": "Amazon.com Inc."},
    {"symbol": "NVDA", "name": "NVIDIA Corp."}, {"symbol": "TSLA", "name": "Tesla, Inc."},
    {"symbol": "META", "name": "Meta Platforms, Inc."}, {"symbol": "BRK-B", "name": "Berkshire Hathaway Inc."},
    {"symbol": "LLY", "name": "Eli Lilly and Company"}, {"symbol": "AVGO", "name": "Broadcom Inc."}
]

# --- Authentication Endpoints ---

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register")
async def register_user(username: str, email: str, password: str, full_name: str | None = None):
    if get_user(username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    hashed_password = get_password_hash(password)
    user_data = {
        "username": username,
        "email": email,
        "full_name": full_name,
        "hashed_password": hashed_password,
        "disabled": False
    }
    users_collection.insert_one(user_data)
    return {"message": "User registered successfully"}

# --- Watchlist Endpoints ---

@app.post("/watchlist/add/{symbol}")
async def add_to_watchlist(symbol: str, current_user: User = Depends(get_current_user)):
    if not any(company["symbol"] == symbol.upper() for company in companies):
        raise HTTPException(status_code=404, detail="Symbol not found")
    
    result = users_collection.update_one(
        {"username": current_user.username},
        {"$addToSet": {"watchlist": symbol.upper()}}
    )
    
    if result.modified_count == 0:
        return {"message": "Symbol already in watchlist"}
    return {"message": f"Added {symbol} to watchlist"}

@app.get("/watchlist")
async def get_watchlist(current_user: User = Depends(get_current_user)):
    user = users_collection.find_one({"username": current_user.username})
    return {"watchlist": user.get("watchlist", [])}

@app.delete("/watchlist/remove/{symbol}")
async def remove_from_watchlist(symbol: str, current_user: User = Depends(get_current_user)):
    result = users_collection.update_one(
        {"username": current_user.username},
        {"$pull": {"watchlist": symbol.upper()}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Symbol not found in watchlist")
    return {"message": f"Removed {symbol} from watchlist"}

# --- API Endpoints ---

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/companies")
async def get_companies():
    return companies

@app.get("/stock/{symbol}")
async def get_stock_data(symbol: str):
    try:
        # Fetch the last year of data (approx 252 trading days)
        cursor = collection.find({"symbol": symbol.upper()}, {"_id": 0}).sort("Date", -1).limit(252)
        data = list(cursor)
        data.reverse() # Ascending order for the chart
        if not data:
            raise HTTPException(status_code=404, detail="Stock data not found.")
        
        # Convert to numpy arrays for technical analysis
        prices = np.array([d['Close'] for d in data], dtype=float)
        
        # Calculate RSI
        rsi = talib.RSI(prices, timeperiod=14)
        
        # Calculate MACD
        macd, macd_signal, macd_hist = talib.MACD(prices)
        
        # Calculate Bollinger Bands
        upper, middle, lower = talib.BBANDS(prices, timeperiod=20)
        
        # Add technical indicators to the data
        for i, d in enumerate(data):
            d['RSI'] = None if np.isnan(rsi[i]) else float(rsi[i])
            d['MACD'] = None if np.isnan(macd[i]) else float(macd[i])
            d['MACD_Signal'] = None if np.isnan(macd_signal[i]) else float(macd_signal[i])
            d['MACD_Hist'] = None if np.isnan(macd_hist[i]) else float(macd_hist[i])
            d['BB_Upper'] = None if np.isnan(upper[i]) else float(upper[i])
            d['BB_Middle'] = None if np.isnan(middle[i]) else float(middle[i])
            d['BB_Lower'] = None if np.isnan(lower[i]) else float(lower[i])
        
        return jsonable_encoder(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stock data: {e}")

# --- NEW ENDPOINT FOR KEY STATS ---
@app.get("/stock/{symbol}/indicators")
async def get_stock_indicators(symbol: str):
    try:
        cursor = collection.find({"symbol": symbol.upper()}, {"_id": 0}).sort("Date", -1).limit(252)
        data = list(cursor)
        if not data:
            raise HTTPException(status_code=404, detail="Indicator data not found.")
        
        high_52_week = max(item['High'] for item in data)
        low_52_week = min(item['Low'] for item in data)
        avg_volume = sum(item['Volume'] for item in data) / len(data)

        return {
            "high_52_week": round(high_52_week, 2),
            "low_52_week": round(low_52_week, 2),
            "average_volume": int(avg_volume)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not calculate indicators: {e}")


# --- (Keep your existing /predict and /sentiment endpoints here) ---
@app.get("/predict/{symbol}")
async def predict_stock_price(symbol: str):
    try:
        cursor = collection.find({"symbol": symbol.upper()}, {"_id": 0}).sort("Date", -1).limit(365)
        data = list(cursor)
        data.reverse()
        if len(data) < 100:
            raise HTTPException(status_code=404, detail="Not enough historical data.")
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        model = ARIMA(df['Close'], order=(5, 1, 0))
        model_fit = model.fit()
        forecast_result = model_fit.get_forecast(steps=7)
        forecast_mean = forecast_result.predicted_mean
        confidence_intervals = forecast_result.conf_int()
        last_date = df.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
        forecast_data = {
            "dates": [d.strftime('%Y-%m-%d') for d in future_dates],
            "predicted_prices": [round(p, 2) for p in forecast_mean],
            "conf_lower": [round(c[0], 2) for c in confidence_intervals.values],
            "conf_upper": [round(c[1], 2) for c in confidence_intervals.values],
        }
        backtest_predictions = model_fit.predict(start=1, end=len(df['Close'])-1)
        backtest_data = {
            "dates": df.index[1:].strftime('%Y-%m-%d').tolist(),
            "predicted_prices": [round(p, 2) for p in backtest_predictions],
        }
        return {"forecast": forecast_data, "backtest": backtest_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not generate prediction: {e}")

@app.get("/sentiment/{symbol}")
async def get_news_sentiment(symbol: str):
    API_KEY = "d2gvv4hr01qon4e9rlb0d2gvv4hr01qon4e9rlbg" 
    if API_KEY == "YOUR_FINNHUB_API_KEY_HERE":
        return {"overall_sentiment": "Neutral", "score": 0, "articles": [{"title": "API Key not configured in app.py", "sentiment": "NEUTRAL"}]}
    today = date.today()
    seven_days_ago = today - timedelta(days=7)
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={seven_days_ago}&to={today}&token={API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        news_data = response.json()
        headlines = [article['headline'] for article in news_data[:5] if article['headline']]
        if not headlines:
            return {"overall_sentiment": "Neutral", "score": 0, "articles": []}
        sentiments = sentiment_pipeline(headlines)
        score = sum(s['score'] if s['label'] == 'POSITIVE' else -s['score'] for s in sentiments)
        avg_score = score / len(sentiments)
        overall_sentiment = "Positive" if avg_score > 0.2 else "Negative" if avg_score < -0.2 else "Neutral"
        articles = [{"title": head, "sentiment": sent['label']} for head, sent in zip(headlines, sentiments)]
        return {"overall_sentiment": overall_sentiment, "score": round(avg_score, 3), "articles": articles}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not fetch news sentiment: {e}")
