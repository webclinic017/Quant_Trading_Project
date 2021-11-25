import pip
import sys
try:
	from statsmodels.stats.contingency_tables import mcnemar
	from api_db_interface import get_historical_data, close, save_path

	import asyncio
	import time

	import torch.nn.functional as F
	import torch.multiprocessing as mp
	import torch
	import torch.nn as nn
except ImportError:
	pip.main(['install', "statsmodels"])
	sys.exit("Installed new modules, please rerun the script")
	
class Net(nn.Module):

    def __init__(self, sa_size):
        super(Net, self).__init__()
        # dropout = 0.1
        # self.lstm1 = nn.LSTM(sa_size, sa_size, batch_first = True)
        self.fc0 = nn.Linear(sa_size, sa_size)
        self.fc1 = nn.Linear(sa_size, 1)
        
    def forward(self, inp):
        # x = F.relu(self.lstm1(inp))
        x = F.relu(self.fc0(inp))
        # x = x + inp
        return self.fc1(x)

# prepare training/test data for neural network
async def prepare_dataset(train_start, train_end):

	dataset = []
	set_of_dates = None
	
	symbols = ["N225", "JPY", "STOXX50E", "EUR", "FTSE", "GBP", "GSPC", "INTEREST"]
	
	for index, item in enumerate(symbols):
		if index == 7:
			data = await get_historical_data(train_start, train_end, "INTEREST")
		else:
			if index % 2 == 0:
				data = await get_historical_data(train_start, train_end, item, ".INDX")
			else:
				data = await get_historical_data(train_start, train_end, item, ".FOREX")	

		dataset.append(data)
		
		if set_of_dates == None:
			set_of_dates = set([bar[0] for bar in data])
		else:
			set_of_dates = set_of_dates.intersection(set([bar[0] for bar in data]))
	

	# make uniform dates and transform the bars into actual inputs
	set_of_dates = sorted(set_of_dates)
	spy_data = []
	for index, date in enumerate(set_of_dates):
		for data_id, data in enumerate(dataset):
			while data[index][0] != date:
				del data[index]
			
			# if INTEREST
			if data_id == 7:
				data[index] = float(data[index][1])
			else:
				if data_id == 6:
					spy_data.append(data[index])
				data[index] = round((data[index][4] - data[index][1])/data[index][1], 4)
				
	# for data in dataset:
		# print(len(set_of_dates), len(data))

	train_x = []
	train_y = []
	
	for index in range(len(set_of_dates)):
		tempi = []
		tempo = []
		for data_id, data in enumerate(dataset):
			if data_id == 6:
				tempo.append(data[index])
			else:
				tempi.append(data[index])
			
		train_x.append(tempi)
		train_y.append(tempo)
		# print(tempi, tempo)
		
	return train_x, train_y, spy_data


# prelim, contains code to ttain neural network (when option is "t") and code to generate mcnemar test (when option is not "t")
async def prelim():
	
	option = ""
	
	if option == "t":
		train_x, train_y, _ = prepare_dataset("2006-01-04", "2011-01-04")
		train_x = torch.Tensor(train_x)
		train_y = torch.Tensor(train_y)
		
		try:

			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
			
			import os.path
			if os.path.isfile(save_path):
				net = torch.load(save_path)
			else:
				net = Net(len(symbols)-1).to(device)
			
			
			
			import torch.optim as optim
			criterion = nn.MSELoss()
			# criterion = nn.CrossEntropyLoss()
			optimizer = optim.Adam(net.parameters())

			
			
			net.train()
			iter_count = 0
			while True:
				optimizer.zero_grad()
				outputs = net(train_x.to(device))
				loss = criterion(outputs, train_y.to(device))
				loss.backward()
				optimizer.step()
				
				if iter_count % 20 == 0:
					print("loss: ", loss.item(), outputs[0], train_y[0])
				iter_count += 1
				

		except KeyboardInterrupt:
			torch.save(net, save_path)
			print('Output saved to: "{}./*"'.format(save_path))		
			
	else:
		test_x, test_y, _ = await prepare_dataset("2011-01-04", "2012-01-05")
		ctable = [[0, 0], [0, 0]]
		net = torch.load(save_path)
		net.eval()

		
		net_correct_count = 0
		benchmark_correct_count = 0
		
		for index, x in enumerate(test_x):
			if index == 0:
				continue

			benchmark_ans = test_y[index - 1][0]
			net_ans = net(torch.tensor(x)).tolist()[0]
			correct_ans = test_y[index][0]
			print(benchmark_ans, net_ans, correct_ans)
			
			# use true because we want to true to be at index 0 and false to be at index 1
			if correct_ans >= 0:
				ctable[not net_ans >= 0][not benchmark_ans >= 0] += 1
				if net_ans >= 0:
					net_correct_count += 1
				if benchmark_ans >= 0:
					benchmark_correct_count += 1			
			else:
				ctable[not net_ans < 0][not benchmark_ans < 0] += 1
				if net_ans < 0:
					net_correct_count += 1
				if benchmark_ans < 0:
					benchmark_correct_count += 1		
		print(ctable)
		print(net_correct_count, benchmark_correct_count)
		print(net_correct_count/len(test_x), benchmark_correct_count/len(test_x))
		
		print(mcnemar(ctable, exact=False, correction=True))


if __name__ == "__main__":
	loop = asyncio.get_event_loop()
	# loop.run_until_complete(loop.create_task(mcnemar_test()))
	loop.run_until_complete(loop.create_task(prelim()))
	close()
