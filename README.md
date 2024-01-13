# NBA Games Prediction Project

Welcome to the NBA Games Prediction project repository! This project focuses on utilizing Data Science and Machine Learning techniques to predict outcomes of NBA games. The repository contains various files and folders to organize the data, code, and processes involved in the project.

## Repository Structure

1. **data**: This folder contains subfolders for different types of data.

   - **CurrentSeason**: Information about the teams' current season games.
   - **TeamsData**: Data from the teams for the previous four seasons.
   - **TeamsPrep**: Data after applying several preprocessing techniques.
   - **TeamsRollings**: Data processed with a window rolling technique.

2. **team_functions.py**: Python file with main functions for use in other files by importing them.

3. **scrapper.ipynb**: Jupyter notebook file where web-scraping techniques are performed to gather data.

4. **scrapper_current.ipynb**: Jupyter notebook file specifically designed to scrape data for the current season.

5. **data_preparation.ipynb**: Jupyter notebook file where all the preprocessing is performed on the previously gathered data.

## How to Use

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/nba-games-prediction.git
   ```
2. Execute the notebooks in the following order:
- scrapper.ipynb
- scrapper_current.ipynb
- data_preparation.ipynb

3. Explore the team_functions.py file for reusable functions and incorporate them into your own scripts or notebooks.
