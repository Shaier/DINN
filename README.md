# DINN
We introduce Disease Informed Neural Networks (DINNs) — neural networks capable of learning how diseases spread, forecasting their progression, and finding their unique parameters (e.g. death rate). Here, we used DINNs to learn the systems of 11 infectious diseases. These systems vary in their complexity, ranging from 3D to 9D ODEs, and from a few parameters to over a dozen. The diseases include COVID, Anthrax, HIV, Zika, Smallpox, Tuberculosis, Pneumonia, Ebola, Dengue, Polio, and Measles. Our contribution is three fold: first, we extend the recent physics informed neural networks (PINNs) approach to a large number of infectious diseases. Second, we perform an extensive analysis of the capabilities and shortcomings of PINNs on diseases. Lastly, we show the ease at which one can use DINN to effectively predict the spread of COVID from real-life data.

<br/><br/>

<p align="center">
   COVID Model</span>
</p>

<div align="center">
   <br/><br/>
  <img src="https://github.com/Shaier/DINN/blob/master/Experiments/real_data/covid_real_data_daily_cases.jpg" width="450" />
  <img src="https://github.com/Shaier/DINN/blob/master/Experiments/real_data/covid_real_data_cumulative_cases.jpg" width="450" /> 
  <br/><br/>
  <pre>
        Daily Cases                                                Cumulative Cases
  <pre>
</div>
