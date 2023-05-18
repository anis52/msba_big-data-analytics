# Application of House Price Prediction Model on Mortgage Loan Scenario


## 1. Executive Summary

House price prediction is of paramount importance in the banking and finance industry. In the US, banks rely largely on the house appraisal form submitted by individual appraisers to decide the mortgage loan. Yet, our research has found that appraisers’ valuation process might be subjective and biased. This inadequacy creates significant risk and potential loss for banks. In this report, we would stand from the banks’ perspective and aimed to develop an objective tool to assess the true house value and address the conflict by reducing the potential risk and loss induced from the inflated appraisal. 

A scientific machine learning method is developed to estimate the house price. We experimented with 6 Machine Learning methods to train our model and XGBoost is adopted as the best model with the lowest training RMSE of 0.0561. Interestingly, we found that ‘GarageCond’ ranks high in the feature importance and this invites further investigation in the next stage.

By comparing with the bank’s status quo and quantitative simulation, our model could help the bank to reduce 40% proportion of risk they were taking, and the amount of extra risk-free money the bank can earn is at least $1,457,312. This proved that our model can significantly mitigate the risk borne by the bank.

## 2. Business Scenario
According to our research, a typical mortgage approval process for the bank in the US involves mainly two parts in the assessment: the consumer’s ability to repay mortgage loans and the actual valuation of the house. For the former, banks would review applicants’ financial situation and calculate applicants’ affordability by scrutinizing their bank account statements and relevant assets proof. This former procedure is relatively standard and reflects applicants’ true repaying ability, whereas the latter might include unknown factors and subjective judgments. 
In the US, the valuation of a house will be performed by an appraiser, who will suggest to the seller, lender, and buyer how much a house is worth. The appraisal is created based on in-person inspection, recent sales of similar properties, current market trends and aspects of the house. However, the price appraised can be largely biased due to several reasons. First is subjectiveness. Research has shown that bias exists in the appraisal process since subjective analysis and imperfect information were involved (Nakamura, 2010[ Nakamura, L. I. (2010). How much is that home really worth? Appraisal bias and house-price uncertainty. Business Review, (Q1), 11-22.]). Second is the pressure from sellers and buyers. In a 2007 study by October Research, a real estate news provider, 90% of more than 1,200 appraisers polled reported feeling pressure to change property values. Over-inflated appraisals occur in the effort to help a seller get more money for the house, or a buyer get a better position to finance mortgage payments. Furthermore, if the home was overpriced, the resale value will not pay for the full price of the original loan. Also, banks can be forced by bond market investors to buy back defaulting mortgages that have inflated appraisals. These undoubtedly create risk for the banking operations and profitability, which is an urgent problem to be solved.

## 3. Conflicts
To ensure a fair appraisal, it is important that the appraiser must be an impartial third party who has no conflict of interest in the outcome of the sale. However, appraisers are hired by lenders, sellers, or buyers. The Center for Public Integrity showed that appraisers who refused to hit the number of estimated values required by the sellers will be put in the “blacklists”. Those who give in to the pressure will work backward from the estimated price to find recent real estate sales that would “make the value”. Since appraising homes is subjective, it was easy to fudge numbers. Therefore, a completely neutral and objective number is urgently needed for banks and buyers to protect themselves from overpaying. 
On the other hand, mortgage as one of the core businesses in the banking industry has always been in a dilemma to balance between approving the applications and the risk of mortgage default. Banks hope to lend to people who can repay, thus it is crucial to alleviate the adverse selection by performing stringent stress testing on the applicants. 
Our sophisticated machine learning model is developed based on 59 variables associated with the characteristics of the house and would provide a more scientific way to predict the house price and hence a better solution for the bank to adopt to calculate the mortgage rate.

##4. Data Exploration and Preprocessing
Data preprocessing is a predominant part of machine learning to yield highly insightful and reliable results. Good data quality increases the accuracy of the model application and leads to better business decision-making. Therefore, we performed a thorough inspection of our data set and implemented corresponding countermeasures to improve data quality. Our data preprocessing steps are divided into data cleaning and data transformation.

###4.1 Outlier Detection 
We treated outliers first because they will introduce bias into the mean of features and affect the result of normalization. To identify outliers, we can plot a scatter diagram and define points that are farthest from the regression line as outliers. From the plot (Figure 1) we can see that there are two dots with ‘GrLivArea’ > 4000 & ‘SalePrice’ < 300000 that fail to fit the pattern. Then we checked the other conditions of these two houses and it turned out that though these are new houses, there are no other significant features that contribute to a low price. Therefore, we can safely conclude that the house with such a large living area is abnormal to have a ‘SalePrice’ below average and dropped these two points and we delete them.

###4.2 Wrong Data Type Transformation
After examining the schema of our dataset, we discovered some features that are labeled incorrectly by the ‘inferSchema’ option. For example, numerical features such as ‘LotFrontage’, ‘MasVnrArea’, and ‘BsmtFinSF1’ should have data type as double instead of string. Also, features like ‘MSSubClass’ and ‘MoSold’ should be categorical instead of numerical. We did the transformation accordingly for better model analysis.

###4.3 Typo Correction & Missing Values Imputation
####4.3.1 Typo Correction
From the description chart of the data, we noted that there is obviously wrong data: GarageYrBlt, which has a max value of 2207. Since the whole data set was collected before 2010, we corrected 2207 to a more reasonable value, 2007.
####4.3.2 Inconsistent Correlated Features
There are some correlated features in this dataset. We can infer their relationships by common sense or by plotting the boxplots. For example, if a house’s ‘MasVnrArea’ has a value, we can infer that the house does have a Masonry veneer and that its ‘MasVnrType’ should also have a value. However, we spotted that there are many inconsistencies in these features. For example, ID 2611 has a ‘MasVnrArea’ of 198 but with a ‘MasVnrType’ of “NA”. For these kinds of issues, we performed the correction and imputation by filling the mode of the entry with the same value of correlated features.
####4.3.3 Other Missing Values
We checked the proportion of missing values in each feature. Generally, we will drop the variable if 40% of its values are missing. However, these missing values in some specific features like ‘PoolQC’, ‘Alley’, ‘Fence’, and ‘FirePlaceQu’ may indicate that this house simply does not have a pool or alley access or a fence or a fireplace. Therefore, in case of misunderstanding, we replaced ‘NA’ with ‘Non’ for categorical variables and 0 for numerical ones to show the real meaning of “NA”. 
‘LotFrontage’ is another feature that has a relatively large number of missing values, so we cannot simply impute it with median value. By plotting the scatterplots and boxplots (Figure 2), we found the features that correlate with ‘LotFrontage’, so we imputed it with these features, including ‘BldgType’, ‘Neighborhood’, using the model of bagged trees.


Figure 2 Features correlated with LotFrontage
For other features that have a small proportion of missing values, we first checked if they are correlated with some other features. If true, we used the mode of the entry with the same value of correlated features. If false, we filled the missing value with mode for categorical and median for numerical features.

###4.4 Near-zero Variance Features Removal 
By definition, zero variance or near-zero variance features are predictors that have only one unique value or that have both of the following characteristics: they have very few unique values relative to the number of samples and the ratio of the frequency of the most common value to the frequency of the second most common value is large. For example, in the training data set, there are 1453 ‘None’ values in total 1460 values of PoolQC, indicating a small variance. Information in these features almost does not change at all and does not warrant any learning from such a feature. Therefore, we can safely remove these features. 

###4.5 New Feature Creation
We noted that there are some time-related and area-related features, so we decided to do some combinations of the existing features and dropped the original ones. We created ‘DurBuilt’, the duration between the house was built and sold, ‘DurGarageBlt’, the duration between the garage was built and sold, and ‘DurRemodAdd’, the duration between the house was remodeled and sold. Likewise, we created ‘AvgRmAbvGrd’ by dividing the size of total above-ground living area by the total number of rooms, and ‘TotalArea’ by aggregating the size of total above-ground living area and size of the basement.

###4.6 Normalization & Standardization
We used the package of recipes in R to perform data normalization & standardization. By plotting the distribution diagram, we can see that most of the variables are positively skewed, so we applied log transformation on them. Then, we centered and scaled all the numerical data to equalize the data variability.

###4.7 Label encoding & One hot encoding
Since most of the algorithms work better with numerical inputs, we transformed our string variables to integer variables. For categorical features that have inherent orders, like ‘LotShape’, ‘ExterQual’, ‘ExterCond’, ‘HeatingQC’, ‘KitchenQual’, we would like to manually encode them to preserve their ordinality. For other categorical features that do not have a specific order, we used StringIndexer and One hot encoder to do transformation.

###4.8 Lumping
For categorical features with high cardinality, after applying one-hot encoder, the dataset will become very wide and sparse, which may harm the computational purposes. Therefore, we sacrifice some of the less frequent values in one feature and lump them together at first.
The complete data description and exploratory data analysis are shown in Appendix 1 & 2.

##5. Machine Learning Questions & Solution
###5.1 ML Question Formulation
We used 6 Machine Learning modeling methods to train our model, which included linear regression (Lasso and Ridge), Random Forest (RF), Support Vector Machines (SVM), Gradient Boosting Machines (GBM), and XGBoost. In the XGBoost model, with limited computational power, we split the tuning process into two stages as the tuning strategy to explore hyperparameter grid search. The first stage searches for max_depth, min_child_weight, and gamma, whereas the second stage is tuning alpha, lambda, and learning rate. Table 1 summarizes the best hyperparameters and RMSE among different machine learning algorithms.
Table 1 ML algorithms results (Grid Search is shown in Appendix 3)
ML Model	Best hyperparameters	Training RMSE
Lasso/Ridge	-	0.11881
Random Forest	maxDepth = 10, numTrees = 500, subsamplingRate = 1, minWeightFraction PerNode = 0	0.05983
SVM	cost = 0.01, gamma = 100, kernel = linear	0.11049
GBM	maxDepth = 3, maxIteration = 100, minNodeSize = 15, learning Rate = 0.1	0.11069
XGBoost	n.estimators = 800, max_depth = 20, min_child_weight = 8, gamma = 0.01, alpha = 0.5 , lambda = 5, learning_rate = 0.05	0.05610

From the above, we saw that the best model we achieved is XGBoost, with the smallest training RMSE of 0.05610. Meanwhile, it reached the smallest score (0.12777) in the Kaggle competition.
5.2 Mortgage Model
To objectively compare our model with the status quo, we adopted a comparison table for objective analysis. 
Table 2 Comparison Table for Objective Analysis
	1. Benchmark Mortgage 	2. Bank (By Appraiser)	3. Our Predicted Model
Sales Price 	Ture Sale Price
(Training Data)	Predicted Sale Price
(Linear regression)	Predicted Sale Price
(XGBoost)
Mortgage (income, loan)	Ideal Mortgage Loan
(More on Appendix 4, 5)	Predicted Mortgage Loan
(More on Appendix 6)	Predicted Mortgage Loan
(More on Appendix 6)

5.2.1 Benchmark Mortgage (Please refer to Appendix 4-5 for all the supporting research)
Firstly, a benchmark mortgage is built with the following assumptions:
1)All calculations are based on a 30-year loans plan
2)Applicants who purchase a more expensive house have a higher income level
3)All applicants have passed the income assessment which proved their ability to repay the loan
The sale price in this benchmark model is extracted from the true sale price in the training data. In order to predict the ideal mortgage amount for each sale, we calculated the income following the average Ames city median disposable income and annual expenditures:

Figure 3 Benchmark Mortgage Calculation
5.2.2 Bank Mortgage (By Appraiser) 
To imitate how the appraiser assesses the house price, we adopted a linear regression model to predict the sale price for comparison. Based on a home appraisal checklist reported by Investopedia, we selected some features that matched the estimators the appraiser used when gauging the house valuation. 
Table 3 Home Appraisal Checklist and Feature Selection
Description	Features
Condition of the home, with specific focus on damage	'OverallQual', 'OverallCond'
Condition of appliances, furnace, air 
conditioning, water heater, and other mechanicals	'HeatingQC', 'CentralAir', 'Electrical'
Size of the home and property	'LotArea', 'totalArea'
Quality of landscaping	'Fence'
Quality of roofing and foundation	'RoofStyle', 'Foundation', 'ExterQual', 'ExterCond'
Number of rooms, bedrooms, closets, bathrooms, and windows	'BedroomAbvGr', 'TotRmsAbvGrd', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath'
Quality of lighting and plumbing, Number of fireplaces	'Fireplaces', 'FireplaceQu'
Quality of the basement, including whether it is finished or unfinished	'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtUnfSF', 'TotalBsmtSF'
Upgrades, remodeling, modernization	'DurBuilt', 'DurRemodAdd'
Similar in size and features to the house	'Neighborhood', 'BldgType', 'LotConfig', 'HouseStyle'

To calculate the mortgage loan, we set a threshold income. If applicants’ annual income is more than the threshold income, the maximum loan is at most 50% of the above predicted house price. Otherwise, the maximum loan is at most 40%. (See Appendix 6)
5.2.3 Our Mortgage with Machine Learning Model 
As mentioned above, we used XGBoost as our model to predict the house price. To compare with the current practice fairly, we will adopt the same mortgage loan calculation method as banks here. 
5.3 Benchmarking & Results
To ensure the effectiveness of our model, we made use of Excel VBA to conduct simulations more than 1,000 times to reflect the impact of the WTP ratio. We also set up an evaluation model as below to quantify our results. The key idea is to compare our predicted mortgage and the bank’s current mortgage, whoever gets a value closer to the benchmark mortgage would be considered as the winner in that entry. Table 5 summarize 4 different scenarios.
Table 5 Win-Loss Comparison

In scenarios 1 and 2, both our model and the bank’s model suggested a lower amount of mortgage loan than the benchmark mortgage loan. We considered this as the safety net, that is the money lent here is considered as having a very low loan default probability. From a bank's perspective, lending more money out can generate higher profit for them in these scenarios. Yet, in scenarios 3 and 4, banks are proposing higher mortgage loans than the benchmark which means banks are on the risky side of lending to people who may not be able to repay the debt. 
Table 6 Win Rate Under Risk Scenario

Under the risky scenarios, our model still has a higher winning rate than the bank’s model. Our winning rate is 67% when both of our models are in a risky scenario. When the bank’s model is at risk, our model’s odd of winning is around 80%. In order words, our model could help the bank to reduce 40% proportion of risk they were taking.
Based on our calculations, on the risky side, the total amount that our model suggested to lend is $281,731 less than the bank’s model. And on the safety side, the total amount that our model suggested to lend out is $408,489 more than the bank’s model. After incorporating with the interest rate effect, the amount of extra risk-free money the bank can earn is at least $1,457,312. Both have proven that our model operates in accordance with bank’s risk preferences and is performing statistically better than the current model that banks are adopting. 
6. Findings and Discussion
6.1 Potential Use
Business analysts can widely adopt our model to interpret different business scenarios efficiently and flexibly. In our paper, we predict the sale price through a scientific way, in which we take a housing sale condition in Ames city as a preliminary example to demonstrate the usage of XGBoost by tuning several hyperparameters. The data preprocessing and hyper grid tuning process takes around 3 working days. The technique will become advanced after testing and trial many times. For instance, banks keep collecting more information about the actual selling price and apartment's qualities, and analysts are able to extract them to improve the model's predicting power. Then, they can minimize their mortgage risk and obtain an ancillary tool to approve for the mortgage applications rather than relying on the third parties’ subjective analysis, thus bringing more unbiased and accurate pricing methods to management. If this model is fully adopted, the user-friendly programming design allows staff in the mortgage department to fill in details of housing conditions and automatically calculate the predicted housing price. Therefore, this prediction provides a reference for determining the mortgage application.
This model is not only applied to the mortgage approval but also implements this technique in enhancing the value stream among the bank’s strategies. When the analysis involves large-scale structured data, XGBoost can get optimal results.
6.2 Benefits to Relevant Stakeholders
As far as banks are concerned, compared to the traditional mortgage approval process, our model can grant more low-risk mortgage loans to homebuyers and provide an additional tool for validating the mortgage loan. It facilitates banks to measure the financial risk in more spectrums and reduces market disinformation. The home appraisal is not the only source for judging the actual housing values and makes the housing market's information more transparent. Our conservative mechanism will approve less high-risk loans and deleverage the company's balance sheet, such as reducing credit loss and debt, which is healthier and stabler than the pre-financial crisis level.
Also, from the housing market perspective, our model can become robust to the banking system's stability and tighten the credit boom. Our model plays a vital role in alleviating credit expansion. Banks will not be over-optimistic for the housing market development and will not lend inappropriate loans to homebuyers. It generally minimizes the possibilities of unprecedented bankruptcy and foreclosure in the banking system. 
In terms of the social perspective, our model promotes good financial management and a suitable debt-to-income ratio. Our model, which is conservative compared to the traditional model, advocates offering 50% of the predicted mortgage loan. Homebuyers receive transparent information of a house's condition and how banks estimate the housing price in a fair and open manner in order to reduce the house price uncertainties. It will provide a message for the homebuyers purchasing a home within their financial ability and keeping a healthy loan-to-income ratio instead of over-leveraging in the investing strategy. 
6.3 Limitations and Further Direction
Our study faces several limitations, namely limited sample size and outdated datasets. Our designated model fits mortgage loan applications in the US rather than global banking businesses. If banks apply this model in other regions’ banking businesses, we will further revise some parameters and formulas. As we found that ‘GarageCond’ ranks high in the feature importance, the future model can lay more emphasis on this feature, which may produce more accurate results. Also, our model may not fit well in the unstructured data such as videos and photos, and other advanced deep-learning methods may be better for analyzing unstructured data.
We do not consider applicants’ credit ratings in the current method due to a lack of personal information. We believe in customizing appropriate mortgage loans that depend on the applicant's income levels and credit ratings to raise our estimated accuracy of an optimal mortgage loan level and borrow more low-risk loans.
