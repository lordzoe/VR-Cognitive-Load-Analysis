##############################################################################
# INTEGRATED R SCRIPT FOR SRANIPAL-UNITY EYE-TRACKING DATA
# Mimicking logic from "Classifying Mental Effort..." code,
# plus the notion of Trials (1..32) and derived Blocks (1..8).
##############################################################################

set.seed(42)

# (Uncomment to install needed packages the first time you run)
# install.packages("lme4")
# install.packages("ggplot2")
# install.packages("eyetrackingR")
# install.packages("readr")
# install.packages("tidyr")
# install.packages("saccades")
# install.packages("zoo")
# install.packages("dplyr")
# install.packages("PupillometryR")
# install.packages("data.table")

#install.packages("gazeR")
library(lme4)
library(ggplot2)
library(eyetrackingR)
library(readr)
library(tidyr)
library(saccades)
library(zoo)
library(dplyr)
library(PupillometryR)
library(data.table)

##############################################################################
# 1) Helper functions for outlier removal & baseline correction
#    (As in the original "Classifying Mental Effort..." code)
##############################################################################

remove_outliers <- function(x, na.rm = TRUE, ...) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
  H <- 1.5 * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- NA
  y[x > (qnt[2] + H)] <- NA
  y
}

baseline_correction <- function(x, median, na.rm = TRUE, ...) {
  m <- median[1]
  y <- x - m
  y
}

##############################################################################
# 2) Define data path and read CSV
##############################################################################

# Define folder with raw eye-tracking data
raw_data_path <- "C:/Users/Mobile Workstation 3/OneDrive - Queen's University/Coding Scripts/EyeTrackingData/Raw_EyeTrackingData"
output_data_path <- "C:/Users/Mobile Workstation 3/OneDrive - Queen's University/Coding Scripts/EyeTrackingData/Processed_EyeTrackingData"

# Find all raw CSV files that match your EyeTrackingData_ParticipantXX_... naming
csv_files <- list.files(path = raw_data_path, pattern = "EyeTrackingData_Participant\\d+_.*\\.csv$", full.names = TRUE)

# Create a data frame to hold block-level results from all participants
all_block_data <- data.frame()

# Loop through each file, process it, and save the cleaned result
for (file in csv_files) {
  
  # Extract participant ID from filename, e.g.: "EyeTrackingData_Participant15_2023_..."
  match <- regexpr("Participant(\\d+)", basename(file), perl=TRUE)
  participant_id <- as.numeric(regmatches(basename(file), match)[1])
  
  # Skip if no valid participant ID found
  if (is.na(participant_id)) next
  
  # Read raw CSV
  rawdata <- read_csv(file, trim_ws=TRUE)

##############################################################################
# 3) Convert timestamps to numeric (milliseconds) for precise time calculations
##############################################################################

  # Convert timestamps to numeric milliseconds
  rawdata$TimestampPOSIX <- as.POSIXct(rawdata$Timestamp, format="%Y-%m-%d %H:%M:%OS")
  start_time <- min(rawdata$TimestampPOSIX, na.rm=TRUE)
  rawdata$Time_ms <- as.numeric(difftime(rawdata$TimestampPOSIX, start_time, units="secs")) * 1000

##############################################################################
# 4) 3D -> 2D projection & scaling to “Tobii-like” coordinates
#    e.g. mapping X,Y in [-1..1] to 0..1920 x 0..1080
##############################################################################

  scale_to_tobii_x <- function(x) {
    # map -1..1 => 0..1920
    return( (x+1)/2 * 1920 )
  }
  scale_to_tobii_y <- function(y) {
    # map -1..1 => 0..1080, flipping Y so +1 => top => 0
    return( (1 - y)/2 * 1080 )
  }

  rawdata <- rawdata %>%
    mutate(
      GazeRightx = scale_to_tobii_x(RightGazeX),
      GazeRighty = scale_to_tobii_y(RightGazeY),
      GazeLeftx  = scale_to_tobii_x(LeftGazeX),
      GazeLefty  = scale_to_tobii_y(LeftGazeY)
    )

##############################################################################
# 5) Define trackloss using BitMask columns (valid=15, invalid=9, etc.)
##############################################################################

  # Define trackloss using BitMask columns (valid=15, invalid=9)
  rawdata <- rawdata %>%
    mutate(
      Trackloss = ifelse(LeftEye_BitMask == 9 & RightEye_BitMask == 9, TRUE, FALSE)
    )
  
  # Rename pupil diameter columns
  rawdata <- rawdata %>%
    rename(
      PupilLeft  = LeftPupilDiameter,
      PupilRight = RightPupilDiameter
    )

  # Trackloss analysis per trial (subset entire trial window)
  trackloss_by_trial <- rawdata %>%
    group_by(Subject, Trial) %>%
    summarise(
      TracklossForTrial = mean(Trackloss, na.rm = TRUE),  # Compute trackloss %
      .groups = "drop"
    )
  
  # Trackloss analysis per block (aggregated across 4 trials)
  trackloss_by_block <- rawdata %>%
    group_by(Subject, Block) %>%
    summarise(
      TracklossForBlock = mean(Trackloss, na.rm = TRUE),  # Compute trackloss %
      .groups = "drop"
    )
  
  # Trackloss analysis per participant
  trackloss_by_subject <- rawdata %>%
    group_by(Subject) %>%
    summarise(
      TracklossForParticipant = mean(Trackloss, na.rm = TRUE),
      .groups = "drop"
    )
  
  # Remove trials where >25% of the data was lost
  trials_to_remove <- trackloss_by_trial %>%
    filter(TracklossForTrial > 0.25) %>%
    pull(Trial)
  
  cleaned_data <- rawdata %>%
    filter(!(Trial %in% trials_to_remove))
  
  # Save cleaned dataset after removing high-trackloss trials
  write.csv(cleaned_data, file.path(output_data_path, paste0("Trackloss_Cleaned_Participant", participant_id, ".csv")), row.names = FALSE)
  
  # Save trackloss summaries
  write.csv(trackloss_by_trial, file.path(output_data_path, paste0("Trackloss_by_Trial_Participant", participant_id, ".csv")), row.names = FALSE)
  write.csv(trackloss_by_block, file.path(output_data_path, paste0("Trackloss_by_Block_Participant", participant_id, ".csv")), row.names = FALSE)
  write.csv(trackloss_by_subject, file.path(output_data_path, paste0("Trackloss_by_Participant", participant_id, ".csv")), row.names = FALSE)
  
  # Trackloss proportion summary
  num_trials_removed <- length(trials_to_remove)
  total_trials <- length(unique(rawdata$Trial))
  proportion_removed <- round((num_trials_removed / total_trials) * 100, digits = 2)
  
  print(paste("Participant", participant_id, ": Removed", num_trials_removed, "trials (", proportion_removed, "%) due to high trackloss.", sep=" "))
  
##############################################################################
# 6) Interpolate missing values & label Subject, Trial, StimuliName
##############################################################################

  # If -1 is used to flag missing data:
  rawdata[rawdata == -1] <- NA
  
  rawdata$GazeRightx <- na.approx(rawdata$GazeRightx, na.rm=FALSE)
  rawdata$GazeRighty <- na.approx(rawdata$GazeRighty, na.rm=FALSE)
  rawdata$GazeLeftx  <- na.approx(rawdata$GazeLeftx,  na.rm=FALSE)
  rawdata$GazeLefty  <- na.approx(rawdata$GazeLefty,  na.rm=FALSE)
  rawdata$PupilRight <- na.approx(rawdata$PupilRight, na.rm=FALSE)
  rawdata$PupilLeft  <- na.approx(rawdata$PupilLeft,  na.rm=FALSE)

  # Assign Subject ID, Trial, Block
  rawdata$Subject <- participant_id
  rawdata$Trial   <- rawdata$CurrentTrial
  rawdata$Block   <- ceiling(rawdata$CurrentTrial / 4)  # 8 blocks total

##############################################################################
# 7) Define your AOIs (two molecules), in “Tobii-like” coords
#    Adjust these bounding boxes carefully if needed.
##############################################################################

  # Define AOIs (molecule left & right) in Tobii-like coordinates
  left_mol_xmin  <- scale_to_tobii_x(-1.85)
  left_mol_xmax  <- scale_to_tobii_x(-0.95)
  left_mol_ymin  <- scale_to_tobii_y( 0.45)
  left_mol_ymax  <- scale_to_tobii_y(-0.45)
  
  right_mol_xmin <- scale_to_tobii_x(0.95)
  right_mol_xmax <- scale_to_tobii_x(1.85)
  right_mol_ymin <- scale_to_tobii_y(0.45)
  right_mol_ymax <- scale_to_tobii_y(-0.45)
  
  aoi_molecules <- data.frame(
    aoi_name = c("molecule_left", "molecule_right"),
    Left     = c(left_mol_xmin,  right_mol_xmin),
    Right    = c(left_mol_xmax,  right_mol_xmax),
    Top      = c(left_mol_ymin,  right_mol_ymin),
    Bottom   = c(left_mol_ymax,  right_mol_ymax)
  )
  
  rawdata_aoi <- add_aoi(
    data = rawdata,
    aoi_dataframe = aoi_molecules,
    x_col = "GazeRightx",
    y_col = "GazeRighty",
    aoi_name = "aoi_name",
    x_min_col = "Left",
    x_max_col = "Right",
    y_min_col = "Bottom",
    y_max_col = "Top"
  )

##############################################################################
# 8) Convert to eyetrackingR object, analyze trackloss, remove Trials >25% lost
##############################################################################

  # Convert to eyetrackingR object, remove Trials >25% trackloss
  data_et <- make_eyetrackingr_data(
    rawdata_aoi,
    participant_column = "Subject",
    trial_column       = "Trial",
    time_column        = "Time_ms",
    trackloss_column   = "Trackloss",
    aoi_columns        = c("molecule_left", "molecule_right"),
    treat_non_aoi_looks_as_missing = FALSE
  )
  
  # Optional: subset_by_window, if your data has message-based time windows:
  # e.g. response_window <- subset_by_window(data, window_start_msg="StartTrial", 
  #                                          msg_col="Event", rezero=TRUE)
  # Then remove trackloss:
  trackloss <- trackloss_analysis(data_et)
  data_clean <- clean_by_trackloss(data_et, trial_prop_thresh=0.25)
  trackloss_clean <- trackloss_analysis(data_clean)
  
  # If you want, write out the cleaned data:
  # write.csv(data_clean, file.path(path, "eyetracking_cleaned.csv"), row.names=FALSE)

##############################################################################
# 9) Detect fixations (2D) with the saccades package
##############################################################################

  # Detect fixations
  samples <- data_clean %>%
    dplyr::select(Time_ms, GazeRightx, GazeRighty, Trial, Block, Subject) %>%
    rename(
      time  = Time_ms,
      x     = GazeRightx,
      y     = GazeRighty,
      trial = Trial
    )
  
  fixations <- detect.fixations(samples, smooth.coordinates=TRUE, smooth.saccades=FALSE)

  # detect.fixations typically strips extra columns, so re-add "Block" via left_join:
  fixations <- fixations %>%
    left_join(samples %>% dplyr::select(trial, Block, Subject), by="trial")

  # AOIs on fixations
  fixations_aoi <- add_aoi(
    data         = fixations,
    aoi_dataframe = aoi_molecules,
    x_col        = "x",
    y_col        = "y",
    aoi_name     = "aoi_name",
    x_min_col    = "Left",
    x_max_col    = "Right",
    y_min_col    = "Bottom",
    y_max_col    = "Top"
  )

##############################################################################
# 10) Summaries by Trial, by Block, by AOI, etc.
##############################################################################

  # Fixations by Trial:
  fixation_by_trial <- fixations %>%
    group_by(trial) %>%
    summarise(
      num_fixations  = n(),
      mean_dur_ms    = mean(duration),
      total_dur_ms   = sum(duration),
      .groups = "drop"
    )
  
  # Fixations by Block:
  fixation_by_block <- fixations %>%
    group_by(Block) %>%
    summarise(
      num_fixations = n(),
      mean_dur_ms   = mean(duration),
      total_dur_ms  = sum(duration),
      .groups = "drop"
    )
  
  # Fixations by AOI & Block:
  fixation_by_aoi_block <- fixations_aoi %>%
    group_by(Block, aoi_name) %>%
    summarise(
      nfix     = n(),
      mean_dur = mean(duration),
      .groups  = "drop"
    )

##############################################################################
# 11) Pupillometry (similar to "Classifying Mental Effort..." approach)
##############################################################################

  pup_data <- data_clean %>%
    select(Subject, Trial, Time_ms, StimuliName, PupilRight, PupilLeft)
  
  data_pup <- make_pupillometryr_data(
    data     = pup_data,
    subject  = Subject,
    trial    = Trial,
    time     = Time_ms,
    condition= StimuliName
  )
  
  regressed_data <- regress_data(data=data_pup, pupil1="PupilRight", pupil2="PupilLeft")
  mean_data      <- calculate_mean_pupil_size(data=regressed_data, pupil1="PupilRight", pupil2="PupilLeft")
  filtered_data  <- filter_data(data=mean_data, pupil="mean_pupil", filter="median")
  
  # Normalize pupil size per participant (z-score standardization)
  final_pupil <- filtered_data %>%
    group_by(Subject) %>%
    mutate(mean_pupil_corrected = scale(mean_pupil))
  
  # Summarize pupil size by Block per participant
  pupil_by_block <- final_pupil %>%
    group_by(Subject, Block) %>%
    summarise(
      mean_pupil_corrected = mean(mean_pupil_corrected, na.rm=TRUE),
      .groups = "drop"
    )

##############################################################################
# 12) Save summary outputs for each participant
##############################################################################
  
  all_block_data <- bind_rows(all_block_data, block_summary)
  
  # (If you also want trial-level detail or other outputs per participant, you could save those too.)
  
  cat("Finished processing participant:", participant_id, "\n")
  
}

##############################################################################
# 13) Save summary outputs for each participant
##############################################################################

write.csv(fixation_by_trial, file.path(output_data_path, paste0("Fixations_by_Trial_Participant", participant_id, ".csv")), row.names=FALSE)
write.csv(fixation_by_block, file.path(output_data_path, paste0("Fixations_by_Block_Participant", participant_id, ".csv")), row.names=FALSE)
write.csv(fixation_by_aoi_block, file.path(output_data_path, paste0("Fixations_by_AOI_Block_Participant", participant_id, ".csv")), row.names=FALSE)
write.csv(pupil_by_block, file.path(output_data_path, paste0("Pupillometry_by_Block_Participant", participant_id, ".csv")), row.names=FALSE)

print(paste("Processed Summaries for Participant", participant_id))

# You could also define "Block = ceiling(Trial / 4)" in that data to group pupil size by block.
# Then summarize as needed.