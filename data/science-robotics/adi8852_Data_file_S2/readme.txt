Data from "Estimating Human Joint Moments Unifies Exoskeleton Control and Reduces User Effort" by Dean D. Molinaro, Inseung Kang, and Aaron J. Young
Dataset prepared by Dean D. Molinaro (contact: dmolinaro3@gatech.edu)
Last Modified: 5/16/2023

This data consists of the p-values and number of participants for each statistical test conducted in this study. The data is stored as a .mat file that can be loaded and visualized in MATLAB using the load() command (i.e., load("stats.mat")).

/////  General Terminology  /////
Ambulation Modes
LG: level ground
RA: ramp ascent
RD: ramp descent
SA: stair ascent
SD: stair descent

Controllers
UC: unified joint moment controller
SC: spline-based controller
NE: not wearing the exoskeleton
ZT: zero torque controller

////  File Contents  /////
stats.mat: The p-values (i.e., p) and number of participants (i.e., n) are provided for all statistical tests run in this study.
The file contains a MATLAB structure array (i.e., struct) with the following structure:
data: the variable name of the struct
	test_1: results for comparing controller effects on metabolic cost during level ground walking (Fig. 2b)
	test_2: results for comparing controller effects on metabolic cost during incline walking (Fig. 2e)
	test_3: results for comparing positive biological joint work during level walking (Fig. 3a)
	test_4: results for comparing positive biological joint work during incline walking (Fig. 3b)
	test_5: results for comparing the TCN RMSE to that of the Baseline method across ambulation modes (Fig. 4a)
	test_6: results for comparing the TCN R2 to that of the Baseline method across ambulation modes (Fig. S6a)
	test_7: results for comparing the TCN RMSE to that of the Baseline method within each ambulation mode (Fig. 4c)
	test_8: results for comparing the TCN R2 to that of the Baseline method within each ambulation mode (Fig. S6c)
	test_9: results for comparing TCN RMSE on conditions within the train set compared to not within the train set for each ambulation mode (Fig. S7a)
	test_10: results for comparing TCN R2 on conditions within the train set compared to not within the train set for each ambulation mode (Fig. S7b)
	test_11: results for comparing TCN RMSE with respect to the training set size (Fig. S8a)
	test_12: results for comparing the TCN R2 with respect to the training set size (Fig. S8b)

/////  Additional Notes  /////
Note: Due to experimental complications, there are a few participants missing specific trial data. Therefore, pairwise comparisons of hip moment estimation accuracy involving the following trials were n = 9 (all other comparisons were n = 10).
	C0p0_S1p9_<any estimator type>
	Cn15p0_S0p75_<any estimator type>
	C12p7_S0p0_holdout
	Cn12p7_S0p0_holdout
