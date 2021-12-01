import pip
import sys

try:
	from prelim_stats import Net
	from api_db_interface import save_path, generic_get, place_order, AP_BASE, close, AP_DATA, live_result_path, proj_folder
	import asyncio

	import torch


except ImportError:
	pip.main(['install', "pandas", "aiohttp", "websocket-client"])
	sys.exit("Installed new modules, please rerun the script")


# prepare input data for neural network
async def prepare_dataset(now):
	tasks = []
	loop = asyncio.get_event_loop()
	
	tasks.append(loop.create_task(generic_get("https://eodhistoricaldata.com/api/real-time/N225.INDX?fmt=json&s=JPY.FOREX,STOXX50E.INDX,EUR.FOREX,FTSE.INDX,GBP.FOREX&api_token=", "eod", proj_folder + "realtime.html")))

	
	forexes = ["JPY", "EUR", "GBP"]
	for symbol in forexes:
		tasks.append(loop.create_task(generic_get("https://eodhistoricaldata.com/api/intraday/" + symbol + ".FOREX?fmt=json&api_token=", "eod", proj_folder + symbol + ".html")))	

	tasks.append(loop.create_task(generic_get("https://markets.newyorkfed.org/read?productCode=50&eventCodes=500&limit=25&startPosition=0&sort=postDt:-1&format=xml", "newyorkfed", proj_folder + "interest.html")))

	
	results = await asyncio.gather(*tasks)
	inputs = []
	
	# first process index 1,2,3 so that we remain with the open bar of forex
	now = now.strftime('%Y-%m-%d')
	for index in range(1,4):
		open_bar = None
		for bar in results[index]:
			if bar["datetime"][0:10] == now:
				open_bar = bar
				break
		results[index] = open_bar
		
		
		
	symbols = ["N225", "JPY", "STOXX50E", "EUR", "FTSE", "GBP"]
	for index, bar in enumerate(results[0]):
		if index % 2 == 0:
			open_price = bar["open"]
		else:
			open_price = results[(index+1)//2]["open"]
			
		latest_price = bar["close"]
		inputs.append(round((latest_price - open_price)/open_price,4))

		print(symbols[index], open_price, latest_price)
			
	inputs.append(results[-1])
	return inputs



'''
async def prepare_dataset(now):
	inputs = []
	
	symbols = ["N225", "JPY", "STOXX50E", "EUR", "FTSE", "GBP"]
	tasks = []
	loop = asyncio.get_event_loop()
	for index, symbol in enumerate(symbols):
		if index % 2 == 0:
			tasks.append(loop.create_task(generic_get("https://eodhistoricaldata.com/api/intraday/" + symbol + ".INDX?fmt=json&api_token=", "eod", folder + symbol + ".html")))
		else:
			tasks.append(loop.create_task(generic_get("https://eodhistoricaldata.com/api/intraday/" + symbol + ".FOREX?fmt=json&api_token=", "eod", folder + symbol + ".html")))		
	
	tasks.append(loop.create_task(generic_get("https://markets.newyorkfed.org/read?productCode=50&eventCodes=500&limit=25&startPosition=0&sort=postDt:-1&format=xml", "newyorkfed", folder + "interest.html")))
	
	tasks.append(loop.create_task(generic_get("https://eodhistoricaldata.com/api/real-time/N225.INDX?fmt=json&s=JPY.FOREX,STOXX50E.INDX,EUR.FOREX,FTSE.INDX,GBP.FOREX&api_token=", "eod", folder + "realtime.html")))
	
	results = await asyncio.gather(*tasks)
		
	now = now.strftime('%Y-%m-%d')
	
	for index, close_bar in enumerate(results[-1]):
		open_price = None
		for bar_id, bar in enumerate(results[index]):
			# print(bar["datetime"][0:10])
			if bar["datetime"][0:10] == now:
				open_price = bar["open"]
				break

		# no open bar today, use close bar yesterday
		if open_price == None:
			print(index, "today's open price not available, using yesterday's close price...")
			open_price = results[index][-1]["close"]
		
		close_price = close_bar["close"]
		print(now, index, open_price, close_price)
		# print(open_price, close_price)
		results[index] = round((close_price - open_price)/open_price,4)		
		
	del results[-1]
	return results
'''


		
# note that alpaca does not support directly going from long to short (and vice versa)< hence this function acts in between live() and place_order()
async def switch_pos(symbol, pos, target_side):

	retry = 0
	while retry < 10:
		
		url = AP_DATA + "/v2/stocks/" + symbol + "/trades/latest"
		latest_trade = await generic_get(url, "alpaca")
		now_price = latest_trade["trade"]["p"]
		# print(now_price)
		
		if pos != 0:
			result = await place_order(symbol, -pos, now_price)
			if not result:
				retry += 1
				time.sleep(1)
				continue
			else:
				print("pos:", pos, "->", 0)
				pos = 0

		url = AP_BASE + "/v2/account"
		account_info = await generic_get(url, "alpaca")
		cash = float(account_info["cash"])
		
		
		target_pos = int(target_side*cash*0.95/now_price)
		result = await place_order(symbol, target_pos, now_price)
		if result:
			print("pos:", pos, "->", target_pos)
			pos = target_pos
			break
		else:
			retry += 1
			time.sleep(1)
			continue
		
			
			
	return pos


from datetime import datetime
from dateutil.tz import gettz
import os
import time
import csv
async def live():

	last_updated = int(os.path.getmtime(save_path))
	net = torch.load(save_path)
	net.eval()
	
	csv_data = []
	pos_response = await generic_get(AP_BASE + "/v2/positions/" + "SPY", "alpaca")
	pos = int(pos_response["qty"])



	while True:
		now = datetime.now(tz=gettz('US/Eastern'))
		open_start = now.replace(hour=9, minute=30, second=0, microsecond=0)
		open_end = now.replace(hour=9, minute=35, second=0, microsecond=0)
		print(now.hour, ":", now.minute, "pos: ", pos, end='\r')
		
		# realtime_x = await prepare_dataset(now)
		# net_ans = net(torch.tensor(realtime_x)).tolist()[0]		
		# print(realtime_x, net_ans)
		# time.sleep(200000)
		
		if now >= open_start and now <= open_end:
			mo_response = await generic_get(AP_BASE + "/v2/clock", "alpaca")
			is_market_open = mo_response["is_open"]
			if is_market_open:
				realtime_x = await prepare_dataset(now)
				pos_changed = False
				account_info = await generic_get(AP_BASE + "/v2/account", "alpaca")
				balance = float(account_info["equity"])
				
				if realtime_x != None:
					net_ans = net(torch.tensor(realtime_x)).tolist()[0]
					print(realtime_x, net_ans)
					new_pos = pos	
					if (pos >= 0 and net_ans < 0):
						new_pos = await switch_pos("SPY", pos, -1)
					elif (pos <= 0 and net_ans > 0):
						new_pos = await switch_pos("SPY", pos, 1)
					# print(realtime_x, net_ans, pos, new_pos)
					if new_pos != pos:
						pos_changed = True
						pos = new_pos
				
				csv_data.append([pos_changed, balance])		
				# once market_open, process it and enter long sleep
				time.sleep(3600)

		elif int(os.path.getmtime(save_path)) > last_updated:
			last_updated = int(os.path.getmtime(save_path))
			net = torch.load(save_path)
			net.eval()
			
		elif len(csv_data) != 0:
			with open(live_result_path, "a") as f:
				writer = csv.writer(f)
				writer.writerows(csv_data)
			csv_data = []
		time.sleep(1)


loop = asyncio.get_event_loop()
loop.run_until_complete(loop.create_task(live()))
close()





