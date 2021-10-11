# DINN
We introduce Disease Informed Neural Networks (DINNs) — neural networks capable of learning how diseases spread, forecasting their progression, and finding their unique parameters (e.g. death rate). Here, we used DINNs to identify the dynamics of 11 highly infectious and deadly diseases. These systems vary in their complexity, ranging from 3D to 9D ODEs, and from a few parameters to over a dozen. The diseases include COVID, Anthrax, HIV, Zika, Smallpox, Tuberculosis, Pneu- monia, Ebola, Dengue, Polio, and Measles.

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

# Getting Started
The easiest way to get started is to first install the necessary packages:

## Setup
For a quick setup follow the next steps:

conda create -n dinn python=3.6

conda activate dinn

git clone https://github.com/Shaier/DINN.git

cd DINN

pip install -r requirements.txt

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Once the packages are install, the next recommendation is to explore the [tutorial.ipynb](tutorial.ipynb) file.

Other than that, the [experiments](https://github.com/Shaier/DINN/tree/master/Experiments) folder has all the experiments I ran for the paper.
The [diseases](https://github.com/Shaier/DINN/tree/master/Diseases) folder has all the diseases DINN was trained on.
