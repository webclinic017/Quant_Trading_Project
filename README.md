# Quantitative Trading Project

## Table of Contents


[Q1: What is this project about?](#Q1)  
[Q2: Can you demonstrate high-level workflow and results?](#Q2)   
[Q3: Can I hear some technical discussions?](#Q3)   
[Q4: What did you learn from this project?](#Q4)   
[Q5: How to reach you?](#Q5)  
[References](#Ref)  


### Q1: What is this project about?<a name="Q1" />
We practice through the standard procedure of quant trading: 

&#8594; pattern recognition/data preparation  
&#8594; statistical testing  
&#8594; backtesting/risk management  
&#8594; real-time deployment. 

 
### Q2: Can you demonstrate high-level workflow and results?<a name="Q2" />

[2.1. pattern recognition/data preparation (*api_db_interface.py*)](#Q2.1)  
[2.2. statistical testing (*prelim_stats.py*)](#Q2.2)   
[2.3. backtesting/risk management (*backtesting.py*)](#Q2.3)   
[2.4. real-time deployment (*live.py*)](#Q2.4)   



###### 2.1. pattern recognition/data preparation (*api_db_interface.py*) <a name="Q2.1" />

We observe that, likely due to globalization, there seems to be a correlation between return of S&P 500 and that of market indices in Japan (Nikkei 225), Europe (Stoxx 50) and Britain (FTSE 100).

We download data from various sources (e.g. alphavantage/eodhistoricaldata), preprocess them and store them in a database.

Once data is ready, we use a ARIMAX(1,0,0) model to quickly confirm our ideas. To avoid collinearity, we perform regressions between SPY 500 and each of Nikkei 225, Stockxx 50, and FTSE 100 separately. 

| Stock | intercept | slope | AR coefficient | Ljung-Box p-value |
| --- | ---| ----| ---| ---|
| Nikkei 225 | -2.447e-05 | 0.2135 | -0.1656 | 0.83|
| Stockxx 50 | 0.0001 | 0.6035 | -0.2271 | 0.75 |
| FTSE 100 | -0.0002 | 0.7577 | -0.4129 | 0.74|

Note that the slopes are not close to 1. But this is fine since they are not close to 0 either and our intention here is not to do the actual trading with these regression models.



###### 2.2. statistical testing (*prelim_stats.py*) <a name="Q2.2" />

Competing models:
1. Neural network (nn): take today's Nikkei 225, FTSE 100, Stockxx 50 and effective federal funds rate data as input and outputs the expected return of S&P 500.
2. Market chaser (mc): the expected return of S&P 500 today equals to the observed one from the previous trading day
	
Now we perform Mcnemar test (with \alpha = 0.01), where a prediction is correct if it has the same sign as the actual return. The data used here is not seen by the neural network during its training.

We obtain the following contingency table and results:

|	|	mc correct |    mc incorrect  |
|----|---|--|
|**nn correct** |94	|		 80 |
|**nn incorrect** |27|			 33|

| | |
|---|---|
|nn correct rate |74.0%|
|mc correct rate |51.5%|
|pvalue     | 4.98e-07|
|test statistic  | 25.27|


###### 2.3. backtesting/risk management (*backtesting.py*) <a name="Q2.3" />

In our strategy, the neural network's raw prediction gets refined through volatility based risk management and produces daily investment decisions.

We simulate transaction costs of Alpaca trading API [\[1\]](#Ref1). A sample run with initial cash $1M produces the following:

![sample_run.png](https://github.com/SmoothKen/Quant_Trading_Project/blob/main/sample_run.png?raw=true)	

where the top chart plots the OHLCV data of S&P 500 and the bottom chart contrasts the performance between our strategy (green) and the buy-and-hold strategy (purple) in terms of equity balance.
	
	
Our risk management pipeline is configurable through a parameter A. The higher the A, the more risk-averse the trader is. The following are backtesting results assuming various risk appetite:


|   | Neutral (A = 0) |  Slightly averse (A = 0.1) | Very averse (A=1) |
| ------- | ------ | -------- |-------- | 
total_trade |99 | 105| 156 | 
win_rate | 83.8% | 77.1%| 45.5% | 
loss_rate | 16.2% | 22.9% | 54.5% |  
avg_return | 5.1% | 4.5% | 1.5% | 
avg_win | 6.2% | 6.0% | 3.5% | 
avg_loss | -1.1% |  -0.7% | -0.1% | 
max_drawdown | -5.0% |  -4.8% | -3.1% | 



###### 2.4. real-time deployment (*live.py*) <a name="Q2.4" />
Ongoing experiment. The algorithm will automatically fetch data and make trades using Alpaca API at 9:30 Eastern time every trading day without traders' attendance.
	
Several safety checks are implemented to insure against hardware/internet failure. This is another aspect of risk management.



### Q3: Can I hear some technical discussions? <a name="Q3" />

[3.1. pattern recognition/data preparation (*api_db_interface.py*)](#Q3.1)  
[3.2. statistical testing (*prelim_stats.py*)](#Q3.2)   
[3.3. backtesting/risk management (*backtesting.py*)](#Q3.3)   
[3.4. real-time deployment (*live.py*)](#Q3.4)   


###### 3.1. pattern recognition/data preparation (*api_db_interface.py*) <a name="Q3.1" />
1. For customization purposes, we directly interact with the REST APIs using aiohttp library.

2. Note that there is a overlap between trading time of LSE, Euronext and NYSE/Nasdaq. Hence while we use end-of-day data for Nikkei 225, we can only use before-NYSE-open data for FTSE 100 and Stockxx 50.

3. During pattern recognition, since we are dealing the stock returns rather than the stock prices, we may wonder if linear regression is sufficient. However here are the Durbin-Watson statistics: 

|Stock |DW statistics |
|---|---|
|Nikkei 225 | 2.331|
|Stockxx 50|2.440|
|FSTE 100|2.808|

Now given the sample size of approximately 200, the critical value (\alpha = 0.05) is 2.24 [\[7\]](#Ref7). Hence we conclude that there is signfiicant autocorrelation in the residual and therefore linear regression is not a suitable model.


4. Also, there are some arguments against Ljung-Box test on ARIMAX models [\[6\]](#Ref6). Hence as safety check, we perform Breusch-Godfrey tests and obtain the following statistics, which do not trigger alarms at \alpha = 0.05.

|Stock | BG statistics | p-value|
|---|---|----|
|Nikkei 225 |12.14| 0.28|
|Stockxx 50|17.44| 0.07|
|FTSE 100|13.32|0.21|



###### 3.2. statistical testing   (*prelim_stats.py*) <a name="Q3.2" />

Notice that we choose Mcnemar test instead of paired t-test [\[2\]](#Ref2). This design choice is made because of preliminary observations that using exact values introduces unnecessary risks.


Despite evaluating classifiers rather than properties of time series itself (e.g. trend), we technically should still consider the time correlation among data. Advanced statistical tests, e.g. [\[3\]](#Ref3), can be used to address this.


###### 3.3. backtesting/risk management (*backtesting.py*) <a name="Q3.3" />

We find that LSTM layers outperforms dense layers and simple recurrent layers. Residual blocks and dropout do not improve the performance significantly. This is consistent with the literature [\[4\]](#Ref4).


We want make the risk management pipeline configurable, Volatility is not a convenient candidate as it requires traders to constantly lookup the current prices. Instead, our choice of coefficient __A__ is a variation of the absolute risk aversion coefficient __a__ [\[5\]](#Ref5), as defined by

![utility_formula.png](https://github.com/SmoothKen/Quant_Trading_Project/blob/main/utility_formula.png?raw=true)

As we can see from the results, although risk-aversion (larger __A__) reduces exposure to large drawdowns, this benefit is offsetted by smaller gains and transaction costs. This is because we frequently decide to exit instead of holding the position on the volatile days.


###### 3.4. real-time deployment (*live.py*) <a name="Q3.4" />
There is a middle step called paper-trading, Our paper-trading results are slightly weaker than backtesting results. This indicates that there are people exploiting the same idea, but not (yet) to the extent that completely diminishes its profitability.

The algorithm, when not trading, will check if the file in which we save the neural network is updated. This allows us to train the neural network in a separate program, making it more convenient than a multiprocessing design.



### Q4: What did you learn from this project? <a name="Q4" />
Among various lessons and techniques,

- Be cautious about pure technical indicators based trading strategies (e.g. RSI, MACD etc). These indicators suffer from low signal-to-noise ratio and non-repeating black swans.

- Efficient market hypothesis makes our life hard. Groups of professional miners make it even harder. Hence, as the old saying goes, if you cannot beat them, join (and learn from) them.

 

### Q5: How to reach you? <a name="Q5" />

Please feel free to contact me through one of the following:

[Keren's Email (link)](mailto:&#107;5&#115;&#104;&#97;&#111;&#64;ucsd&#46;&#101;&#100;&#117;)  
[Keren's Linkedin (link)](https://www.linkedin.com/in/keren-shao)

### References <a name="Ref" /> 
\[1\]<a name="Ref1" />  [Pricing and fees - Alpaca Forum](https://forum.alpaca.markets/t/pricing-and-fees/2309)  
\[2\]<a name="Ref2" />  [McNemar’s test or T-test for measuring statistical significance -  
Stackexchange](https://stats.stackexchange.com/questions/20013/mcnemar-s-test-or-t-test-for-measuring-statistical-significance-of-matched-pre-p)  
\[3\]<a name="Ref3" />  [Pesaran, M. Hashem, and Allan Timmermann.   
“A Simple Nonparametric Test of Predictive Performance.”  
Journal of Business & Economic Statistics, vol. 10, no. 4 ](https://www.jstor.org/stable/1391822)  
\[4\]<a name="Ref4" />  [Sezer, Omer Berat  
"Financial time series forecasting with deep learning: A systematic literature review: 2005–2019."   
Applied Soft Computing 90 (2020): 106181.](https://arxiv.org/pdf/1911.13288.pdf)  
\[5\]<a name="Ref5" />  [Rao, Ashwin "Understanding Risk-Aversion through Utility Theory"](https://web.stanford.edu/class/cme241/lecture_slides/UtilityTheoryForRisk.pdf)
\[6\]<a name="Ref6" />  [Testing for autocorrelation: Ljung-Box versus Breusch-Godfrey - Stackexchange](https://stats.stackexchange.com/questions/148004/testing-for-autocorrelation-ljung-box-versus-breusch-godfrey)
\[7\]<a name="Ref7" /> [Durbin-Watson Significance Tables](https://www3.nd.edu/~wevans1/econ30331/Durbin_Watson_tables.pdf)
