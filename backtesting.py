import pip
import sys

try:
	from prelim_stats import prepare_dataset, Net
	from api_db_interface import backtest_result_path, save_path, close
	import asyncio
	
	import torch

except ImportError:
	pip.main(['install', "pandas", "aiohttp", "websocket-client"])
	sys.exit("Installed new modules, please rerun the script")

import math
def risk_mgt(inputs, net_ans, threshold):
	
	# compute volatility
	inputs = [inputs[0], inputs[2], inputs[4]]
	mean = sum(inputs)/len(inputs)
	temp = 0
	for item in inputs:
		temp += (item - mean)**2
	volatility = math.sqrt(temp/len(inputs))
	
	# compute sharpe ratio and produce decision
	sharpe = abs(net_ans)/volatility
	if sharpe >= threshold:
		return net_ans
	else:
		return 0
	

async def backtesting():
	test_x, test_y, spy_data = await prepare_dataset("2011-01-05", "2012-01-05")
	
	import os.path
	if os.path.isfile(save_path):
		net = torch.load(save_path)
	else:
		net = Net(len(symbols)-1).to(device)
	net.eval()	
	
	# 1. cannot daytrade
	# 2. transaction cost
	initial_price = spy_data[0][1]
	initial_balance = 1000000
	cash = initial_balance
	pos = 0
	
	csv_data = []
	risk_averse_coeff = 1
	
	for index, bar in enumerate(spy_data):
		
		now_price = bar[1]
		net_ans = net(torch.tensor(test_x[index])).tolist()[0]
		net_ans = risk_mgt(test_x[index], net_ans, risk_averse_coeff)
		
		
		balance = cash + pos*now_price

		if net_ans > 0 and pos <= 0:
			# cover
			pos = balance/now_price
			cash = 0
			pos_changed = True
			
		elif net_ans < 0 and pos >= 0:
			# short, 
			old_pos = pos
			pos = -balance/now_price
			delta = pos - old_pos
			transaction_cost = round(delta*now_price*5.1/1000000 + delta*0.000119, 2)
			cash = 2*balance + transaction_cost
			pos_changed = True

		elif net_ans == 0 and pos < 0:
			pos = 0
			cash = balance
			pos_changed = True	
		
		elif net_ans == 0 and pos > 0:
			old_pos = pos
			pos = 0
			delta = pos - old_pos
			transaction_cost = round(delta*now_price*5.1/1000000 + delta*0.000119, 2)
			cash = balance + transaction_cost
			pos_changed = True	
					
		else:
			pos_changed = False
		
		etf_balance = initial_balance/initial_price*now_price
		print(balance, etf_balance)
		csv_data.append(list(bar) + [pos_changed, balance, etf_balance])
		
	import csv
	with open(backtest_result_path, "w") as f:
		writer = csv.writer(f)
		writer.writerow(["time", "open", "high", "low", "closed", "volume", "pos_changed", "balance", "etf_balance"])
		writer.writerows(csv_data)

loop = asyncio.get_event_loop()
# loop.run_until_complete(loop.create_task(mcnemar_test()))
loop.run_until_complete(loop.create_task(backtesting()))
close()
