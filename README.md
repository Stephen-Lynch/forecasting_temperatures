# Forecasting Rise In Temperatures


 With temperatures slowly on the rise and global warming on the forefront of every nations mind, I wanted to take a shot at making some visualizations to show just how much temperatures have risen in the past 10 years as well as predict what the temperatures may look like in the years to come for some of the worlds hottest cities.

 I started out with a dataset from the United Nations which had the average temperature of any given day for cities all over the world ranging from the years 1995 to 2020. Orignally the beginning of this data is very sparse, with many countries not recording their average daily temperature so I had to fix that up for the earlier years. Other than that, this was an extremely clean dataset to work with!

 ![image](images/Data.png)


 ## Initial Data Analysis ##
 ------------------------------

 ![image](images/Rising_Temps.png)
 
 To start off, I first wanted to see if I could spot some sort of general rising trend in temperatures over the years. Based just on this graph, it seems really hard to tell. In general there does seem to be some sort of rise, however it's not nearly as large of a rise as I was initially expecting. I also find it interesting to see that there does seem to be an almost random spike downwards and then back up around every 3 years or so. While I found this graph interesting, I wanted to double check and make sure there was some sort of general trend and performed a seasonal decompose on the monthly averages.
 
 
![image](images/Seasonal_Decompose.png) 

 This graph really surprised me, even with the data beforehand, I was expecting a much clearer looking general trend than the one we ended up getting in here. It still seems as if there is some sort of slow rise in the peaks, but it looks to be only a 1 to 2 degree difference even starting from 1995.

 Going forward, I decided it would be best to instead look at the cities I would be choosing for my models to see if I noticed anything happening with them as well.
 
 ![image](images/Rising_Temps_3_cities.png)
 
 One thing I found interesting about these cities is that their average yearly temperature hasn't gone up by much. There are certain spikes to be certain and the majority of them seem to be on some sort of upwards trend since 1995, however they're relatively flat lined at least compared to how I originally thought in terms of temperature increases. One thing to note here however is that every single one of these cities has a yearly average of over 80 degrees. Being from Colorado, it's hard to imagine living in a place that has weather that's so hot that it's yearly average temperature is something I would consider toasty here. Many of these places get much hotter than this, such as Niamey, Nigeria hitting a max just this year of 120 degrees Fahrenheit

 With all of this in mind, I wanted to move on and start working on my model. After doing some research on which machine learning model would perform best, I ended up deciding to apply a SARIMAX model to each of the cities, as it seemed to handle seasonality and general trends extremely well.


 ## SARIMAX Models ##
 ------------------------------

 ![image](images/Predictions.png)

 It looks like our model doesn't seem to be predicting as well I as I would hope. It seems to be sticking to some sort of trend far too much despite my efforts trying to make the data stationary. This isn't going to stop me from creating a forecrasting graph however.

 ![image](images/forecast.png)
 
 At this point I just wanted to see what the model thinks is going to happen a few months into the "future" to see how far off it's performing in the long run. The answer is quite a bit. I'm sort of impressed at how well it tends to mimic the peaks and valleys, but it's super far off in terms of predicting anything meaningful for a forecast.
 

 ## Future Steps: ##
 The first thing I would like to do for this project going forward is get a better understanding of what the SARIMAX models in general. As it stands right now, I feel like I have a beginner level of knowledge on the model, which clearly wasn't enough to combat the underfitting of my data. If I did get this model working however, there are few things I wanted to try out. The first of course would be being able to show my original goal. I would also love to show other cities, such as ones in the U.S to make it more relateable for people here.

 ## Conclusions: ##
 My goal orginally with this project was to try and show how much temperatures are rising by showing just how hot the world's hottest cities are getting. While there does seem to be some sort of general rising trend in temperatures across the world, it's hard to tell just how much it really is rising. My models seem to be able to predict based off that rising trend somewhat, but it's just too small to make a massive difference in temperature based on the data I performed my models on.