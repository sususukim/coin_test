import os
from dotenv import load_dotenv
load_dotenv()
import pyupbit
import pandas as pd
import pandas_ta as ta
import json
from openai import OpenAI
import schedule
import time

#upbit & openAI 호출
client = OpenAI(api_key=os.getenv("OPEN_API_KEY")) 
upbit = pyupbit.Upbit(os.getenv("UPBIT_ACCESS_KEY"), os.getenv("UPBIT_SECRET_KEY"))

#전체와꾸는 2시 21분쯔음
def get_current_status():
    orderbook = pyupbit.get_orderbook(ticker="KRW-BTC") #호가창의 실제 오더북 매수/매도북을 가져옴
    current_time = orderbook['timestamp'] #시간대를 맞춤 오더북과
    btc_balance = 0
    krw_balance = 0
    btc_avg_buy_price = 0
    balances = upbit.get_balances()
    for b in balances:                     #현재 내 잔고를 받아옴
        if b['currency'] == "BTC":
            btc_balance = b['balance']
            btc_avg_buy_price = b['avg_buy_price']
        if b['currency'] == "KRW":
            krw_balance = b['balance']
    #이걸 다 받아와서 저장해서 넘겨줌
    current_status = {'current_time': current_time, 'orderbook': orderbook, 'btc_balance': btc_balance, 'krw_balance': krw_balance, 'btc_avg_buy_price': btc_avg_buy_price}
    return json.dumps(current_status)


def fetch_and_prepare_data(): #30일+24시간 데이터 가져옴, 특히나 기술지표들도 Json형태로 가져옴 
    # Fetch data
    df_daily = pyupbit.get_ohlcv("KRW-BTC", "day", count=30)  #한달치
    df_hourly = pyupbit.get_ohlcv("KRW-BTC", interval="minute60", count=24) #하루치

    # Define a helper function to add indicators  #기술지표들 추가 
    def add_indicators(df):
        # Moving Averages  #이런것에 대해 전부 각자 설명을 적어줘야함. 
        df['SMA_10'] = ta.sma(df['close'], length=10)
        df['EMA_10'] = ta.ema(df['close'], length=10)

        # RSI
        df['RSI_14'] = ta.rsi(df['close'], length=14)

        # Stochastic Oscillator
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
        df = df.join(stoch)

        # MACD
        ema_fast = df['close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_fast - ema_slow
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']

        # Bollinger Bands
        df['Middle_Band'] = df['close'].rolling(window=20).mean()
        # Calculate the standard deviation of closing prices over the last 20 days
        std_dev = df['close'].rolling(window=20).std()
        # Calculate the upper band (Middle Band + 2 * Standard Deviation)
        df['Upper_Band'] = df['Middle_Band'] + (std_dev * 2)
        # Calculate the lower band (Middle Band - 2 * Standard Deviation)
        df['Lower_Band'] = df['Middle_Band'] - (std_dev * 2)

        return df

    # Add indicators to both dataframes
    df_daily = add_indicators(df_daily)
    df_hourly = add_indicators(df_hourly)

    combined_df = pd.concat([df_daily, df_hourly], keys=['daily', 'hourly'])
    combined_data = combined_df.to_json(orient='split') #하루, 한달치를 모두 합쳐서 하나의 data로 통합

    # make combined data as string and print length
    print(len(combined_data))

    return json.dumps(combined_data)

def get_instructions(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            instructions = file.read()
        return instructions
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print("An error occurred while reading the file:", e)

def analyze_data_with_gpt4(data_json):
    instructions_path = "instructions.md"
    try:
        instructions = get_instructions(instructions_path)
        if not instructions:
            print("No instructions found.")
            return None

        current_status = get_current_status()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": data_json},
                {"role": "user", "content": current_status}
            ],
            response_format={"type":"json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in analyzing data with GPT-4: {e}")
        return None

def execute_buy():
    print("Attempting to buy BTC...")
    try:
        krw = upbit.get_balance("KRW")
        if krw > 5000:  #최소거래량이 5천원이라서..시장가로 매수
            result = upbit.buy_market_order("KRW-BTC", krw*0.9995)
            print("Buy order successful:", result)
    except Exception as e:
        print(f"Failed to execute buy order: {e}")

def execute_sell():
    print("Attempting to sell BTC...")
    try:
        btc = upbit.get_balance("BTC")
        current_price = pyupbit.get_orderbook(ticker="KRW-BTC")['orderbook_units'][0]["ask_price"]
        if current_price*btc > 5000:
            result = upbit.sell_market_order("KRW-BTC", btc)
            print("Sell order successful:", result)
    except Exception as e:
        print(f"Failed to execute sell order: {e}")

def make_decision_and_execute():  #실제 실행하는 함수. 
    print("Making decision and executing...")
    data_json = fetch_and_prepare_data()  #위에 30+24시간데이터
    advice = analyze_data_with_gpt4(data_json) #이걸 이제 Instruction..아까 길게쓴 .md를 반영해서 분석..그래서 advice가 나옴

    try:
        decision = json.loads(advice) #json으로 받은 조언에 따라 결정
        print(decision)
        if decision.get('decision') == "buy":
            execute_buy()
        elif decision.get('decision') == "sell":
            execute_sell()
    except Exception as e:
        print(f"Failed to parse the advice as JSON: {e}")

if __name__ == "__main__": #결국 얼마마다 한번 거래할것인가? hour은 시간마다 
    make_decision_and_execute()
    schedule.every().hour.at(":01").do(make_decision_and_execute) #사용법은  https://foss4g.tistory.com/1626 여기있음. 

    while True:
        schedule.run_pending()
        time.sleep(1)

#끄는건 ctrl+c
#결국 instruction에 채권관련 대가들의 이야기..등을 가져올수 있게해놓고
#체크/인포 API에서 선물가격 바로 가져올수있게 할까? 