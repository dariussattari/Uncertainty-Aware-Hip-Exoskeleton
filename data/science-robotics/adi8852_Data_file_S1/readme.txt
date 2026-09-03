Data from "Estimating Human Joint Moments Unifies Exoskeleton Control and Reduces User Effort" by Dean D. Molinaro, Inseung Kang, and Aaron J. Young
Dataset prepared by Dean D. Molinaro (contact: dmolinaro3@gatech.edu)
Last Modified: 9/16/2023

This dataset consists of the processed results presented in this study. The dataset consists of five .mat files that can be loaded and visualized in MATLAB using the load() command (e.g., load("est_accuracy.mat")).

/////  General Terminology, Conventions, and Units  /////
Ambulation Modes
LG: level ground (treadmill)
RA: ramp ascent (treadmill)
RD: ramp descent (treadmill)
SA: stair ascent (overground staircase)
SD: stair descent (overground staircase)
STAND: standing
S2W: stand-to-walk transition (treadmill)
W2S: walk-to-stand transition (treadmill)

Controllers
UC: unified joint moment controller
SC: spline-based controller
NE: not wearing the exoskeleton
ZT: zero torque controller

Conventions
Extension/Plantarflexion: positive 
Flexion/Dorsiflexion: negative
Each results struct contains a "condition" field. Conditions are denoted as "C<condition>" to describe the slope (in degrees) or stair height (in cm) of the trial. 
	NOTE: If the trial does not consist of a ramp slope or stair height, <condition> = C0p0.
Each results struct also contains a "speed" field. Speeds are denoted as "S<speed>" to describe the walking speed.
	NOTE: For overground trials, <speed> = S0p0 since the walking speed varied based on the participant.
Condition and speed field names use "p" as a substitute for "." (e.g., 12p5 = 12.5) and "n" as a substitute for "-" (e.g., n12p5 = -12.5) so that the condition and speeds names are valid MATLAB struct field names. 

Units (Unless Otherwise Stated)
Exo Torque: Nm
Joint Power: W/kg
Joint Work: J/kg
Metabolic Cost: W/kg
RMSE: Nm/kg
R2: unitless

////  File Contents  /////
met.mat: Steady-state metabolic cost and corresponding exoskeleton torques are provided for each subject per ambulation mode included in the metabolic cost analysis. These results were used to generate Fig. 2.
The file contains a MATLAB structure array (i.e., struct) with the following structure:
data: the variable name of the struct
	ambulation mode: describes the corresponding ambulation mode
		condition: describes the corresponding slope
			speed: describes the corresponding speed
				controller: describes the corresponding controller. Note: standing data is included as its own controller (i.e., STAND), containing the basal metabolic rate for each subject
					ssMet: a table containing the steady-state metabolic cost for each subject
					ssMetNet: a table containing the steady-state metabolic cost of walking for each subject (i.e., net metabolic cost computed from subtracting STAND.ssMet)
					trq_cmd: the stride-average torque commanded to the actuators for each subject starting at heel strike
						avg: the average of the commanded torque over the stride for each subject
						std: the standard deviation of the commanded torque over the stride for each subject
					trq_mea: the stride-average "measured" torque computed from the measured current of the actuators for each subject starting and ending at heel strike
						avg: the average of the measured torque over the stride for each subject
						std: the standard deviation of the measured torque over the stride for each subject

work.mat: Joint-level power and positive mechanical work computed for each subject per ambulation mode included in the joint work analysis. These results were used to generate Fig. 3.
The file contains a MATLAB structure array (i.e., struct) with the following structure:
data: the variable name of the struct
	ambulation mode: describes the corresponding ambulation mode
		condition: describes the corresponding slope
			speed: describes the corresponding speed
				controller: describes the corresponding controller
					power: contains stride-average power curves for the hip, knee, and ankle starting and ending at heel strike
						<joint>_net: the total power at the joint computed from the joint kinematics and moments resulting from the biomechanical analysis
						<joint>_bio: the power of the biological joint computed after subtracting the exoskeleton assistive torque provided to the joint. Note: <joint>_bio and <joint>_net are equivalent when there is no exo torque for a specified joint.
						<joint>_exo: the exoskeleton power delivered to the biological joint computed based on the exoskeleton torque and the biological joint kinematics. Note: <joint>_exo is zero when there is no exo torque for a specified joint.
							avg: the stride-average power for each subject
							std: the standard deviation of the power over the stride for each subject
					work_pos: contains the stride-average positive mechanical work for the hip, knee, and ankle
						<joint>_net: the total positive work at the joint computed by integrating the _net power time series data.
						<joint>_bio: the positive work done by the biological joint computed by integrating the _bio time series data.
						<joint>_exo: the positive work done by the exoskeleton computed by integrating the _exo time series data.
						total_net: the total positive work across the hip, knee, and ankle, computed by summing their corresponding _net results
						total_bio: the positive work done by the user's hip, knee, and ankle not including the exoskeleton, computed by summing their corresponding _bio results 
						total_exo: the positive work done by the exoskeleton. Note: This is equivalent to hip_exo since the exoskeleton only assisted the hip joint
							avg: the stride-average positive work for each subject
							std: the standard deviation of the positive work for each subject

est_accuracy.mat: The RMSE and R2 of the TCN (implemented online) and Baseline method (implemented offline) are provided for each subject per ambulation mode, condition, and speed. These results were used to generate Figs. 4 and S6.
The file contains a MATLAB structure array (i.e., struct) with the following structure:
data: the variable name of the struct
	estimator: contains results for the TCN (i.e., tcn) or the Baseline Method (i.e., baseline)
		ambulation mode: describes the corresponding ambulation mode
			condition: describes the corresponding slope or stair height of the trial
			ALL: a results table containing the average results for each subject across all conditions for the specified ambulation mode
				speed: a table containing the average results for each subject for the specified ambulation mode, condition, and speed
				ALL: a results table containing the average results for each subject across all speeds for the specified ambulation mode and condition

The results tables have the fields:
subject: a cell array containing the subject corresponding to each row
rmse: the root-mean-square error (RMSE) corresponding to each subject
r2: the R2 corresponding to each subject

est_generalization.mat: The RMSE and R2 of the TCN when tested on the conditions and speeds within the training set (i.e., hold_in) and not within the training set (i.e., hold_out). These results were used to generate Fig. S7. 
The file contains a MATLAB structure array (i.e., struct) with the same structure as est_accuracy.mat but with estimator fields: hold_in and hold_out.

est_trainingsize.mat: The RMSE and R2 of the TCN when trained use the datasets from Phase 1 (i.e., p1), Phases 1 and 2 (i.e., p12), Phases 1, 2, and 3 (i.e., p123), and Phases 1, 2, 3, and 4 (i.e., p1234). These results were used to generate Fig. S8.
The file contains a MATLAB structure array (i.e., struct) with the same structure as est_accuracy.mat but with estimator fields: p1, p12, p123, and p1234.
