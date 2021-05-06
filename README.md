# FYP-BSS-Rebalancing

This is the final year project for the completion of a BSc in computer science. Author Thomas Igoe.

This project involves the prediction of station population in an example bike sharing system (in this case, Dublin Bikes).
These predictions are done with a LSTM recurrent neural network. This information is then used to list "rebalancing jobs".
These jobs are recommended relocation jobs for drivers who work for a BSS, ensuring that the system is balanced.

The implementation and reasoning behind the project are further explored in the corrisponding report pdf.

##  Usage

This project runs on a [python venv](https://docs.python.org/3.8/library/venv.html "Virtual environment documentation")
in python 3.8. The necessary libraries are listed in `requirements.txt`. To use, [set up a python venv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/ "Python venv instructions")
and install `requirements.txt` with the command 
`pip install -r requirements.txt`. When using the jupyter-notebooks, the notebooks will have to be set to your newly created venv. To use the flask frontend, run `frontend/website_view.py`.

## Attribution

The Phoenix park hourly weather station data, "[Historical weather data](https://www.met.ie/climate/available-data/historical-data)"
from [Met Éireann](https://www.met.ie/) is licensed under CC BY [4.0](https://creativecommons.org/licenses/by/4.0/)

"[Dublin Bikes historical data  ](https://data.smartdublin.ie/dataset/dublinbikes-api)"
from [Smart Dublin](https://smartdublin.ie/) is licensed under [Creative commons](http://opendefinition.org/licenses/cc-by/)

## Acknowledgements

Many thanks to Professor Tahar Kechadi for his guidance and input during the development of this project.