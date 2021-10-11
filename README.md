# DINN
We introduce Disease Informed Neural Networks (DINNs) — neural networks capable of learning how diseases spread, forecasting their progression, and finding their unique parameters (e.g. death rate). Here, we used DINNs to identify the dynamics of 11 highly infectious and deadly diseases. These systems vary in their complexity, ranging from 3D to 9D ODEs, and from a few parameters to over a dozen. The diseases include COVID, Anthrax, HIV, Zika, Smallpox, Tuberculosis, Pneu- monia, Ebola, Dengue, Polio, and Measles. Our contribution is three fold. First, we extend the recent physics informed neural networks (PINNs) approach to a large number of infectious diseases. Second, we perform an extensive analysis of the capabilities and shortcomings of PINNs on diseases. Lastly, we show the ease at which one can use DINN to effectively learn COVID’s spread dynamics and forecast its progression a month into the future from real-life data.

<br/><br/>

<p align="center">
   Disease Informed Neural Network Sample Architecture</span>
</p>

<div align="center">
<!--    <br/><br/> -->
  <img src="https://github.com/Shaier/DINN/blob/master/DINN_Sample_Architecture.png" width="680" /> 
<!--   <br/><br/> -->
</div>

<br/><br/>

<p align="center">
   COVID Model: 1 Month Future Predictions </span>
</p>

<div align="center">
<!--    <br/><br/> -->
  <img src="https://github.com/Shaier/DINN/blob/master/Experiments/real_data/covid_real_data_cumulative_cases.jpg" width="680" /> 
<!--   <br/><br/> -->
</div>
