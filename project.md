<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

## My Project

In this project, I applied machine learning techniques to explore and predict Reference Evapotranspiration (Eto), using in-situ data from a CIMIS station. CIMIS stands for California Irrigation Management Information System, and according to their website, it can be described as a program within the Water Use and Efficiency Branch of the California Department of Water Resources (DWR), that oversees a network of more than 145 automated weather stations across California. It was established in 1982 through a collaboration between DWR and the University of California, Davis (UC Davis), CIMIS was created to help irrigation managers optimize water resource usage. By promoting efficient water use, the program helps conserve water, energy, and financial resources for Californians.

Therefore, given my background as a Civil Engineer currently working in irrigation and water resources management, learning Machine Learning techniques will help me find solutions to water scarcity problems more efficiently. 


## Introduction 

Evapotranspiration (ET) refers to the water lost to the atmosphere through the combined processes of evaporation from soil and plant surfaces and transpiration from plant tissues. Reliable ET estimates are essential for multiple purposes. In fields such as agricultural and landscape irrigation, these estimates are highly important for system design, irrigation scheduling, water rights, water transfers, water resources planning, and other water management concerns (California Irrigation Management Information System, CIMIS, 2024). In this context, there is Reference Evapotranspiration (ETo), which is derived by measuring weather conditions and estimating the ET of a reference plant. In California this is a standardized planted surface of well-maintained cool season turf (UCANR, 2024).

Nonetheless, in developing countries, measuring ETo is not always possible, therefore, we rely on Machine Learning (ML) techniques to calculate Eto. In other words, in many regions, in-situ data on weather parameters such as temperature, humidity, and wind speed may be scarce or difficult to obtain. Machine learning models can be trained using available data from nearby weather stations or remote sensing, helping to fill in these data gaps, providing another way to estimate ETo, enabling better water management and irrigation practices without the need for a dense network of physical sensors. 

The primary goal was to develop a model that could accurately predict Eto, a key factor in agriculture and water management. By using historical data, I trained two machine learning models to identify patterns and relationships between these variables. I specifically used supervised learning techniques because the dataset includes both input variables (precipitation, solar radiation) and a target variable (Eto). This allowed me to apply regression methods to predict the target variable based on known inputs.


## Data

Here is an overview of the dataset, how it was obtained and the preprocessing steps taken, with some plots!

![](assets/IMG/datapenguin.png){: width="500" }

*Figure 1: Here is a caption for my diagram. This one shows a pengiun [1].*

## Modelling

Here are some more details about the machine learning approach, and why this was deemed appropriate for the dataset. 

<p>
When \(a \ne 0\), there are two solutions to \(ax^2 + bx + c = 0\) and they are
  \[x = {-b \pm \sqrt{b^2-4ac} \over 2a}.\]
</p>

The model might involve optimizing some quantity. You can include snippets of code if it is helpful to explain things.

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
clf.predict([[0, 0, 0, 0]])
```

This is how the method was developed.

## Results

Figure X shows... [description of Figure X].

## Discussion

From Figure X, one can see that... [interpretation of Figure X].

## Conclusion

Here is a brief summary. From this work, the following conclusions can be made:
* first conclusion
* second conclusion

Here is how this work could be developed further in a future project.

## References
[1] DALL-E 3

[back](./)

