SEPwC Landslide Risk Coursework (Python)
[ENV00046I-S2-A] Solving Environmental Problems with Code

This project trains a machine learning model to predict the probability of landslides across a region from terrain, geological and land cover data, and outputs the result as a probability raster (GeoTIFF) with values between 0 and 1.

The original assignment brief is in the file ASSIGNMENT_INSTRUCTIONS.md.

Running

To run the code, use this command:

python3 terrain_analysis.py --topography data/AW3D30.tif --geology data/geology_raster.tif --landcover data/Landcover.tif --faults data/Confirmed_faults.shp data/landslides.shp probability.tif

The -v or --verbose flag prints the model accuracy and feature importances. The --plot flag also saves a PNG quick-look of the probability raster next to the GeoTIFF.

Tests can be run with pytest from the project root.

How the code works

The main function in terrain_analysis.py runs the full pipeline. It loads the topography, geology and land cover rasters (reprojecting the geology and land cover ones onto the topography grid), derives a slope raster and a distance-to-faults raster, samples all five layers at the landslide points (positives) and at the same number of random points (negatives), trains a RandomForestClassifier on the resulting table, and then predicts a landslide probability for every pixel and writes it out as a GeoTIFF.

Use of AI

AI tools (Google's Gemini, Open AI's Chat GPT and Anthropic's Claude) were used during this assessment, both as a learning aid and to assist in writing parts of the code.

Acknowledgements

This repository is a fork of the SEPwC landslides coursework template provided by the module convenor, Jon Hill (https://github.com/jhill1).
