import pip

try:
	from datetime import datetime
	from dateutil.relativedelta import relativedelta
	from dateutil import tz
	
	import time
	import json
	import aiohttp
	import asyncio
	import sqlite3
	
	import sys
	
	from api_access import *
	
except ImportError:
	pip.main(['install', "aiohttp"])
	print("Installed new modules, please rerun the script")
	sys.exit(0)


proj_folder = "/home/ken/Downloads/"
save_path = proj_folder + "trade_model.h5"
backtest_result_path = proj_folder + "backtest_result.csv"
live_result_path = proj_folder + "live_result.csv" 

AP_BASE	="https://paper-api.alpaca.markets"
AP_DATA = "https://data.alpaca.markets"

alpaca_headers = {"APCA-API-KEY-ID": AP_ID, "APCA-API-SECRET-KEY": AP_SECRET}

session = aiohttp.ClientSession()
db = sqlite3.connect(proj_folder + "historical_data.db")
cur = db.cursor()



# note; INJECTION attack possible. Hence not to be exposed to user input
async def get_historical_data(start, end, symbol, suffix = None):
	
	pkey = "date"
	
	if symbol != "INTEREST":
		cur.execute("CREATE TABLE IF NOT EXISTS " + symbol + "(" + pkey + " primary key, open, high, low, close, volume)")
	else:
		cur.execute("CREATE TABLE IF NOT EXISTS " + symbol + "(" + pkey + " primary key, rate)")


	
	cur.execute("SELECT * FROM " + symbol + " WHERE " + pkey + "=(SELECT max(" + pkey + ") FROM " + symbol + ")")
	last_record = cur.fetchall()
	if len(last_record) == 0:
		http_start = "1970-01-01"
	else:
		end_of_data = last_record[-1][0]
		if end_of_data < end:
			# regardless whether end_of_data is before or after start, fetch from end_of_data to preserve consecutiveness
			http_start = end_of_data
		else:
			cur.execute("SELECT * FROM " + symbol + " WHERE " + pkey + " >= ? and " + pkey + " <= ?", (start, end))
			return cur.fetchall()
	
	
	
	if symbol != "INTEREST":
		url = "https://eodhistoricaldata.com/api/eod/" + symbol + suffix + "?api_token=" + EOD_KEY + "&fmt=json&from=" + http_start + "&to=" + end
	else:
		url = "https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&interval=daily&apikey=" + AV_KEY
		
	print(url)
	data_list = []
	
	async with session.get(url) as response:
		print("historical api Status:", response.status)
		html = await response.text()
		html = json.loads(html)


		if symbol != "INTEREST":
			for bar in html:
				data_list.append([bar["date"], bar["open"], bar["high"], bar["low"], bar["close"], bar["volume"]])
				# print(bar)
			cur.executemany("INSERT OR IGNORE INTO " + symbol + " (" + pkey + ", open, high, low, close, volume) VALUES (?,?,?,?,?,?)", data_list)
		else:
			for bar in html["data"]:
				data_list.append([bar["date"], bar["value"]])
			
			cur.executemany("INSERT OR IGNORE INTO " + symbol + " (" + pkey + ", rate) VALUES (?,?)", data_list)
		db.commit()
		
		
		
		cur.execute("SELECT * FROM " + symbol + " WHERE " + pkey + " >= ? AND " + pkey + " <= ?", (start, end))
		return cur.fetchall()







import xml.etree.ElementTree as ET
async def generic_get(url, api = None):
	if api == "alpaca":
		headers = alpaca_headers
	elif api == "eod":
		url += EOD_KEY
		headers = None
	else:
		headers = None
	
	async with session.get(url, headers=headers) as response:
		html = await response.text()
		if response.status == 200:
			if api != "newyorkfed":
				return json.loads(html)
			else:					
				return float(ET.fromstring(html)[0][1][2].text)
		elif response.status == 404 and url.startswith(AP_BASE + "/v2/positions/"):
			return {"qty": 0}
		else:
			sys.exit(html)
		

async def place_order(symbol, qty, limit_price):

	url = AP_BASE + "/v2/orders"
	body = {"symbol": symbol, "qty": abs(qty), "side": "buy" if qty > 0 else "sell",  "type": "limit", "time_in_force": "fok", "limit_price": limit_price}

	# print(url)	
	# print(body)

	async with session.post(url, headers=alpaca_headers, data=json.dumps(body).encode('utf-8')) as response:
		html = await response.text()
		if response.status == 200 and json.loads(html)["status"] == "accepted":
			# succeed in sending, now tracking till fail or succeed
			track_url = AP_BASE + "/v2/orders/" + json.loads(html)["id"]
			while True:
				
				result = await generic_get(track_url, "alpaca")
				status = result["status"]
				print("waiting....", status)
				
				# only exit when some exit signal reached, do not use e.g. NOT "new"
				if status == "filled":
					print(status)
					return True
				elif status == "canceled" or status == "rejected" or status == "expired":
					print(status)
					return False
				time.sleep(1)

		else:
			print(html)
			return False
 
			



def close():
	loop = asyncio.get_event_loop()
	loop.run_until_complete(loop.create_task(session.close()))
	db.close()



if __name__ == "__main__":
	loop = asyncio.get_event_loop()
	# bars = loop.run_until_complete(loop.create_task(get_historical_data("2010-01-01", "2010-01-04", "JPY", ".FOREX")))
	# bars = loop.run_until_complete(loop.create_task(generic_get("https://eodhistoricaldata.com/api/real-time/N225.INDX?fmt=json&s=JPY.FOREX,SX5E.INDX,EUR.FOREX,FTSE.INDX,GBP.FOREX&api_token=", "eod")))
	bars = loop.run_until_complete(loop.create_task(generic_get(AP_BASE + "/v2/account", "alpaca")))
	# bars = loop.run_until_complete(loop.create_task(place_order("SPY", -1, 500)))
	print(bars)
	close()
