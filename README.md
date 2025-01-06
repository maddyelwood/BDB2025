## Motion Metrics: Tight End Motion Impact on Rushing Lanes 
This repository contains code and analysis for a project submitted to the Undergraduate Track of the 2025 NFL Big Data Bowl Competition. The full detailed report including insights, processes, and results can be accessed here: https://www.kaggle.com/code/madalynelwood/motion-metrics-te-motion-impact-on-rushing-lanes

### Abstract
In modern football, pre-snap motion is often used to create mismatches and exploit defensive coverage schemes. This project investigates the impact of pre-snap motion, specifically by Tight Ends, by analyzing changes in defender spacing, lane geometry, and motion patterns. Using advanced spatial techniques, such as the shoelace formula for area calculations, we quantify key metrics like lane size and changes in spacing. These features are incorporated into predictive models to assess their influence on play success, defined using expected points added (EPA), binary EPA (positive or negative), and yards per carry. By leveraging player tracking data from the NFL 2022 season, this analysis provides insights into the effectiveness of tight end motion on space creation and play success. 

### Data
The data used in this project is sourced from the NFL and PFF tracking datasets, which provide spatial coordinates and metadata for players on the field for each play. The analysis focuses on specific subsets of plays involving pre-snap motion by tight ends, with additional filtering for coverage type (man or zone) and run plays. For more information on the dataset, visit the Kaggle competition page: https://www.kaggle.com/competitions/nfl-big-data-bowl-2025/overview 

### Features 
This repository includes:
- Preprocessing scripts to unify data from multiple CSV files and extract relevant plays.
- Lane detection using the shoelace formula to compute the area of running lanes.
- Defender spacing analysis to calculate spacing differences between defenders pre-motion, post-motion, and throughout the play.
- Statistical and machine learning techniques to evaluate the impact of space creation on play success.

## How to Use
1. Clone this repository.
2. Preproccess NFL tracking data using the scripts provided as well as the lane detection programs for given coverage and dataset types.
3. Use the motion_success_model.py file to train and evaluate models.
4. Visualize results and adapt the analysis for your use case.

## Acknowledgements
This project is a submission to the Kaggle NFL Big Data Bowl Competition, and we extend our gratitude to the NFL for providing access to detailed player tracking data.
