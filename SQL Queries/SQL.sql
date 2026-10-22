CREATE DATABASE telco_churn;


select count(*)
from customers;


select Churn,
	   COUNT(*) as customer_count,
       ROUND(COUNT(*) * 100.0 / (select COUNT(*) from customers), 2) as percentage
from customers
group by Churn;


select contract, count(*) total_customers,sum(case when churn='yes' then 1 else 0 end) as churned_customers,
round(sum(case when churn='yes' then 1 else 0 end) * 100 / count(*), 2) as churn_rate_percent
from customers
group by contract
order by churn_rate_percent desc;


select contract, count(*) total_customers, round(avg(tenure), 1) as avg_tenure_months
from customers
group by contract
order by avg_tenure_months;


select InternetService, count(*) total_customers,sum(case when churn='yes' then 1 else 0 end) as churned_customers,
round(sum(case when churn='yes' then 1 else 0 end) * 100 / count(*), 2) as churn_rate_percent
from customers
group by InternetServiceS
order by churn_rate_percent desc;


select InternetService, count(*) total_customers, round(avg(MonthlyCharges), 2) as avg_monthly_charges
from customers
group by InternetService
order by avg_monthly_charges desc;


select TechSupport, count(*) total_customers,sum(case when churn='yes' then 1 else 0 end) as churned_customers,
round(sum(case when churn='yes' then 1 else 0 end) * 100 / count(*), 2) as churn_rate_percent
from customers
group by TechSupport
order by churn_rate_percent desc;

#===============================================================

select Contract, InternetService, TechSupport, count(*) total_customers,
	   sum(case when churn='yes' then 1 else 0 end) as churned_customers,
	   round(sum(case when churn='yes' then 1 else 0 end) * 100 / count(*), 2) as churn_rate_percent
from customers
group by Contract, InternetService, TechSupport
having total_customers >= 30
order by churn_rate_percent desc;

