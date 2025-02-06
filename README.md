
# BTM 4700 - Machine Learning in Business Analytics




## Syllabus Description
Machine learning (ML) extracts meaningful insights from raw data to quickly solve complex, data-rich business problems. Organizations generate a large volume of structured and unstructured data that needs to be analyzed. Machine Learning can increase the power of decision-making by extracting insights from large data. In this course, students will learn the core concepts and different techniques of machine learning and examine how they can be used to improve business decision-making. Using R, students will apply machine learning techniques on large business datasets. Students will learn the application of machine learning algorithms in multiple business domains (e.g., Healthcare, Supply Chain, and Banking) to gain useful business insights

## Outcomes/Skills
- Data Mining
- Data Cleaning
- Statistical Techniques
- Use Data to Make Informed Decisions
- Feature Extraction

## Project 1 - Real Estate

In this game we are given a set of 4 random independent variables: lot size, number of rooms, number of floors, and number of baths. We are then given an input field to insert what we would sell the house for. Once submitted we would receive either **Sold** or **Unsold**. If our house was sold, we would receive a 6% commission. You had 30 days to make as much commission value as possible.

My first thought was to try and find any rules possible for the game, so I created a script to run through the game a ton of times using a MLPRegressor. The neural network had two hidden layers, one with 64 neurons, the other with 32. We take in our independent variables, then try to predict the best output. I also set aside 10% of the data as a validation set to prevent overfitting of the model.

When it comes to training the model, I only wanted to use successful data. If we found a combination that resulted in significant commission value, I pasted it five times to achieve a greater bias, as this could be ran as many times as I wanted.

In terms of predicting the price, it was a mix of historical data as well as the neural networks. As I got similar data, the model's confidence improved. I always wanted to make sure that it wouldn't just find a number it knew would work so I made a 15% range of where the model could explore when it was performing well, but limited it to 5% if it was performing average.
I retrained the model every 10 runs (300 day simulations).

At 100,000 data points I performed an EDA on my data and was quite annoyed to find that a lot of my results varied and there were no clear rules, but seeing the graphs were still cool.

I eventually found my sweet spot where I could set a baseline value and base a multiplier off the combination of values. Because this was a lot less GPU intensive because this next step didn't require training model, I knew I would be able to run this program a ton, so I added a risk value multiplier to try and stretch my luck.

My final score was $407,231.77, which was **1.9** standard deviations above the mean.
