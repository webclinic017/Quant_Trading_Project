import pip
import sys

try:
	import pandas as pd
	from api_db_interface import backtest_result_path, close
	import asyncio

except ImportError:
	pip.main(['install', "pandas", "aiohttp"])
	sys.exit("Installed new modules, please rerun the script")




# print statistics based on records from backtesting.py and live.py
def print_stats(df):

	wins = []
	losses = []
	initial_balance = 1000000
	
	for index, row in df.iterrows():
		if row["pos_changed"]:
			if index != 0:
				end_balance = row["balance"]
				if end_balance > start_balance:
					wins.append((end_balance - start_balance)/initial_balance)
				else:
					losses.append((end_balance - start_balance)/initial_balance)
				
			# end of one trade is the beginning of another
			start_balance = row["balance"]

	total_trade = len(wins) + len(losses)
	print("total_trade", total_trade)
	print("win_rate", '{:.1%}'.format(len(wins)/total_trade))
	print("loss_rate", '{:.1%}'.format(len(losses)/total_trade))
	print("avg_return", '{:.1%}'.format((sum(wins) + sum(losses))/total_trade))
	print("avg_win", '{:.1%}'.format(sum(wins)/len(wins)))
	print("avg_loss", '{:.1%}'.format(sum(losses)/len(losses)))
	# print("max_win", '{:.1%}'.format(max(wins)))
	print("max_drawdown", '{:.1%}'.format(min(losses)))
	
	


# print candlestick plot given OHLCV data
def candlestick_plot(df):
	import plotly.graph_objects as go
	from plotly.subplots import make_subplots

	# Create figure with secondary y-axis
	fig = make_subplots(rows=2,
		cols=1,
		shared_xaxes=True,
		vertical_spacing=0.02,
		specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
	
	# include candlestick with rangeselector
	fig.add_trace(go.Candlestick(x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["closed"], yaxis='y1', name='S&P 500 OHLC'), secondary_y=True, row=1, col=1)

	# include a go.Bar trace for volumes
	fig.add_trace(go.Bar(x=df["time"], y=df["volume"], yaxis='y2', name='S&P 500 Volume', marker_color = 'lightblue'), secondary_y=False, row=1, col=1)


	fig.add_trace(go.Scatter(x=df["time"], y=df["balance"], mode='lines', name='nn strategy'), row=2, col=1)
	fig.add_trace(go.Scatter(x=df["time"], y=df["etf_balance"], mode='lines', name='buy-and-hold'), row=2, col=1)

	fig.update_layout(xaxis = dict(type="category"), xaxis_rangeslider_visible=False)
	fig.update_layout(bargap=0.0, bargroupgap=0.0)

	fig.show()




df = pd.read_csv(backtest_result_path)
# print(df)
print_stats(df)
candlestick_plot(df)
close()
