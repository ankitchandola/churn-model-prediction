# churn-model-prediction
Employee churn can be defined as a leak or departure of an intellectual asset from a company or organization. Alternatively, in simple words, you can say, when employees leave the organization is known as churn. Another definition can be when a member of a population leaves a population, is known as churn. In Research, it was found that employee churn will be affected by age, tenure, pay, job satisfaction, salary, working conditions, growth potential and employee’s perceptions of fairness. Some other variables such as age, gender, ethnicity, education, and marital status, were essential factors in the prediction of employee churn. In some cases such as the employee with niche skills are harder to replace. It affects the ongoing work and productivity of existing employees. Acquiring new employees as a replacement has its costs such as hiring costs and training costs. Also, the new employee will take time to learn skills at the similar level of technical or business expertise knowledge of an older employee. Organizations tackle this problem by applying machine learning techniques to predict employee churn, which helps them in taking necessary actions.
“Employee Churn Prediction”


 

   INTRODUCTION

•	Definition Data mining is defined as the process of discovering patterns in data. The process must be automatic or (more usually) semiautomatic. The patters discovered must be meaningful in that they lead to some advantage, usually an economic advantage. The data is invariably present in substantial quantities. Data mining is a field, connecting the three worlds of Databases, Artificial Intelligence and Statistics. The information age has enabled many organizations to gather large volumes of data. However, the usefulness of this data is negligible if “meaningful information” or “knowledge” cannot be extracted from it. Data mining, otherwise known as knowledge discovery, attempts to answer this need. In contrast to standard statistical methods, data mining techniques search for interesting information without demanding a priori hypotheses. As a field, it has introduced new concepts and algorithms such as association rule learning. It has also applied known machine-learning algorithms such as inductive-rule learning (e.g., by decision trees) to the setting where very large databases are involved. Data mining techniques are used in business and research and are becoming more and more popular with time.
 Data mining is the process of extracting patterns from data. As more data are gathered, with the amount of data doubling every three years, data mining is becoming an increasingly important tool to transform these data into information. It is commonly used in a wide range of profiling practices, such as marketing, surveillance, fraud detection and scientific discovery. Data mining is used for a variety of purposes in both the private and public sectors. Industries such as banking, insurance, medicine, and retailing commonly use data mining to reduce costs, enhance research, and increase sales. For example:
 • The insurance and banking industries use data mining applications to detect fraud and assist in risk assessment (e.g., credit scoring). 
• Using customer data collected over several years, companies can develop models that predict whether a customer is a good credit risk, or whether an accident claim may be fraudulent and should be investigated more closely. • The medical community sometimes uses data mining to help predict the effectiveness of a procedure or medicine. Pharmaceutical firms use data mining of chemical compounds and genetic material to help guide research on new treatments for diseases. 
• Retailers can use information collected through affinity programs (e.g., shoppers’ club cards, frequent flyer points, contests) to assess the effectiveness of product selection and placement decisions, coupon offers, and which products are often purchased together. 
• Companies such as telephone service providers and music clubs can use data mining to create a “churn analysis,” to assess which customers are likely to remain as subscribers and which ones are likely to switch to a competitor.





II. RELATED WORK
  We investigated to find out how much work is being done in the field of churn analysis. In our search we found other activities that discuss churn analysis in detail. We describe a few functions here.
 In [5] the authors describe the predictive modeling of churner based on data mining methods. This paper also discusses how to use the tree analysis model in detail. This paper focused on customer outreach from a business perspective. However at the end of the paper they also discussed the lessons on process flow and modeling techniques. Vol. 1 No. 19-27, 2010.
 In [6] Teu Mutanen described a study of customer appeal cases. This paper describes in detail the methods used to predict, the data used and the results obtained. The author described two methods of churn analysis. The first is the reversal of order. An asset deficit is used to predict a different outcome depending on the ongoing variance and / or segmentation. In this way there can be only one variation that depends. This method works with a maximum probability ratio after a variable-dependent conversion into a systematic variable. The second method analyzes the estimates of the order of magnitude. It is known as the lifting curve. This curve is related to the ROC curve of signal acquisition theory as well as the precision memory curve. Elevation is the average of the predictive model calculated as the ratio between the results obtained outside the predictive model. April 12, 2014
 In [7] Shamam V. Nath describes a case study in which Oracle based database of 50,000 customers in the wireless telecommunications industry was analyzed to predict churners. The study used JDeveloper's tools and the analysis was performed using the Naïve Bayes algorithm for supervised learning. April 14, 2014
Marco Richeldi and Alessandro Perucci [8] wrote a paper on the study of churn analysis cases. This paper discusses the use of Mining Mart, a churn analysis tool. It talks a lot about data analytics analysis with Mining Mart. April 12, 2014






III. METHODOLOGY

A. Our Data: The data we gathered are of a renowned telecom company.
     To load the dataset we use pandas csv function.
•	This dataset has 14,999 samples, and 10 attributes (6 integer, 2 float, and 2 objects).
•	No variable column has null/missing values.
    The 10 attributes are as follows:
•	satisfaction level: It is employee satisfaction point, which ranges from 0-1.
•	last evaluation: It is evaluated performance by the employer, which also ranges from 0-1.
•	number projects: How many numbers of projects assigned to an employee?
•	average_monthly_hours: How many average numbers of hours worked by an employee in a month?
•	time_spent_company: time_spent_company means employee experience. The number of years spent by an employee in the company.
•	work_accident: Whether an employee has had a work accident or not.
•	promotion_last_5years: Whether an employee has had a promotion in the last 5 years or not.
•	Departments: Employee's working department/division.
•	Salary: Salary level of the employee such as low, medium and high.
•	left: Whether the employee has left the company or not.

 

Data Visualization

Subplots using Seaborn
 It could be time consuming to plot graphs by 1 single attribute so we use   Seaborn library and plot all the graphs in a single run using subplots.

 

V Building a Prediction Model

Pre-Processing Data
Lots of machine learning algorithms require numerical input data, so we need to represent categorical columns in a numerical column.
In order to encode this data, we can map each value to a number. e.g. Salary column's value can be represented as low:0, medium:1, and high:2.
This process is known as label encoding, and sklearn conveniently will do this for us using Label Encoder.
 


Split Train and Test Set
To understand model performance, dividing the dataset into a training set and a test set is a good strategy.
Let's split dataset by using function train_test_split(). We need to pass 3 parameters features, target, and test_set size. Additionally, we can use random_state to select records randomly.

  
 
 
Model Building
Let's build employee a churn prediction model.
Here, we are going to predict churn using Gradient Boosting Classifier.
First, import the GradientBoostingClassifier  module and create Gradient Boosting classifier object using GradientBoostingClassifier() function.
Similarly import RandomForestClassifier() module and create random forest classifier object using RandomForestClassifier() function.
Then, fit we model on train set using fit() and perform prediction on the test set using predict().
Evaluating Model Performance
We got classification rate of 97% in gradient boost and 98.3% in random forest.
Precision: Precision is about being precise, i.e., how precise your model is. In other words, we can say, when a model makes a prediction, how often it is correct. In our prediction case, when our Gradient Boosting model predicted an employee is going to leave, that employee actually left 95% of the time or 99% in case of random forest model.
Recall: If there is an employee who left present in the test set and our Gradient Boosting model can identify it 92% of the time or 93.8% in case of random forest model
VI. The Obstacles We faced
1) Incomplete Data: The data we worked with was incomplete. Therefore, the result we obtained is not accurate. However, it does give us an idea of how churn analysis works.
2) Time: Churn Analysis is a long and complex process. Therefore, the time frame was not sufficient to perform a full churn analysis and to obtain a valid result.
 3) Privacy Issues: A key issue that arises in any data collection is that it is confidential. The need for confidentiality is sometimes caused by law (e.g., of medical information) or may be motivated by business interests. However, there are cases where data sharing can lead to profitability. The key to using great knowledge today is research, whether scientific or economic and market-oriented. So, for example, the medical field has a lot to gain by compiling research data; as can even competing businesses with similar interests. While there are potential benefits, this is often not possible due to the privacy issues that arise









VII Conclusion

What Did We Learn From Employee Churn? Churn's analysis of data mining operations is a critical issue for many programs. Various strategies may play an important role in this domain. However, this paper highlights some of the challenges these strategies face in churn analysis. It has been shown that under certain circumstances it is easy to violate privacy protections provided by different strategies. It has provided comprehensive test results with a variety of data and has shown that this is indeed a concern we should look at. In addition to raising these concerns this paper offers a churn analysis model that can find broader performance in creating a new perspective on developing better churn analysis methods. It is interesting from the company's point of view whether the leading customers are worth keeping or not. Also, from a marketing standpoint it can be done to keep them. The length of the data should also be a matter of interest.
  Scope of this project
The result we got is not promising. This is because we had an incomplete database. Our failure is that we were unable to manage the appropriate database for performing churn analysis. In the future there are many things we want to do for this project. Some of these are listed below -
 • Use a complete database to use the churn analysis method.
 • Use multiple methods to analyze churn.
• Compare different methods to find one.
What we will try to do next is find out how to hide data to maintain privacy. Factors related to the choice of privacy algorithm are: features of a good face mask, risk of disclosure, and, minimal risk of disclosure.




VIII  CITATION

1.	https://www.datacamp.com/community/tutorials/predicting-employee-churn-python
2.	https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Employee+Churn+in+Python/HR_comma_sep.csv
3.	http://en.wikipedia.org/wiki/Supervised_learning
4.	http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
5.	 K. B. Oseman, S.B.M. Shukor, N. A. Haris, F. Bakar, "Data Mining in the Churn Analysis Model for Telecommunication Industry", Journal of Statistical Modeling and Analytics, Vol. 1 No. 19-27, 2010.
6.	T. Mutanen, Customer Analysis churn - case study, Technical Report, Retrieved from, http://www.vtt.fi/inf/julkaisut/muut/2006/customer_churn_ case_study.pdf , case_study.pdf, April 12, 2014
7.	S.V. Nath, Churn Client Analysis in the Wireless Industry: Data Mining Method, Technical Report, received from http://download.oracle.com/owsf_2003/40332.pdf , April 14, 2014. 
8.	M. Richeldi and A. Perrucci, "Churn Analysis Case Study", Technical Report, Telecom Italia Lab, Italy, accessed  http://sfb876.tudortmund.de/PublicPublicationFiles/richeldi_perrucci_200 2b.pdf , April 12, 2014.

             
