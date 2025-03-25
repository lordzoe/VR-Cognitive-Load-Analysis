###############################################################################
# Load Required Packages
###############################################################################
# install.packages("data.table")    # for fast I/O
# install.packages("dplyr")         # for data manipulation
# install.packages("zoo")           # for rollmedian, na.locf, etc.
# install.packages("saccades")      # for detect.fixations()
# install.packages("PupillometryR") # for pupil data handling

library(data.table)
library(dplyr)
library(zoo)
library(saccades)
library(PupillometryR)


###############################################################################
# Define File Paths
###############################################################################
filtered_folder <- "C:\\Users\\Mobile Workstation 3\\OneDrive - Queen's University\\Coding Scripts\\EyeTrackingData\\Filtered_EyeTrackingData"

# Get all .csv files in that folder
files <- list.files(path = filtered_folder, pattern = "\\.csv$", full.names = TRUE)

###############################################################################
# 1. Read & Process Each File: Angle Computation + Rolling Median
###############################################################################
for (file_path in files) {
  
  message("Processing file: ", file_path)
  
  # Read CSV into data.table (fast)
  df <- fread(file_path, stringsAsFactors = FALSE)
  
  #-----------------------------------------------------------------
  # (A) Compute the angles in degrees
  #    Make sure your columns match (LeftGazeX, LeftGazeZ, etc.)
  #    If you have a different naming scheme, adapt accordingly.
  #-----------------------------------------------------------------
  if (!all(c("LeftGazeX","LeftGazeZ","RightGazeX","RightGazeZ",
             "FixationPointX","FixationPointZ") %in% colnames(df))) {
    warning("Some required gaze columns not found in ", file_path)
    next
  }
  
  # atan2(y, x) is the usual convention for angles, but your Python code
  # used atan2(X, Z). We replicate that for consistency:
  df[, LeftEx := atan2(LeftGazeX, LeftGazeZ) * 180 / pi ]
  df[, RightEx := atan2(RightGazeX, RightGazeZ) * 180 / pi ]
  df[, CombinedEx := atan2(FixationPointX, FixationPointZ) * 180 / pi ]
  
  #-----------------------------------------------------------------
  # (B) Rolling Median with window size = 10
  #    R has multiple ways: runmed() in base R, rollmedian() in zoo, etc.
  #    Below are two examples. Uncomment whichever approach you prefer:
  #-----------------------------------------------------------------
  
  ## Example 1: Use zoo::rollmedian
  df[, Filtered_LeftEx := rollmedian(LeftEx, 10, fill=NA, align="center")]
  df[, Filtered_RightEx := rollmedian(RightEx, 10, fill=NA, align="center")]
  df[, Filtered_CombinedEx := rollmedian(CombinedEx, 10, fill=NA, align="center")]
  
  ## Example 2: Use base R runmed() [Uncomment to try it instead]
  # df$Filtered_LeftEx     <- runmed(df$LeftEx,     k=10, endrule="median")
  # df$Filtered_RightEx    <- runmed(df$RightEx,    k=10, endrule="median")
  # df$Filtered_CombinedEx <- runmed(df$CombinedEx, k=10, endrule="median")
  
  # If you want to fill the NA edges, you can do something like:
  df[, Filtered_LeftEx := na.locf(na.locf(Filtered_LeftEx, fromLast=TRUE), fromLast=FALSE)]
  df[, Filtered_RightEx := na.locf(na.locf(Filtered_RightEx, fromLast=TRUE), fromLast=FALSE)]
  df[, Filtered_CombinedEx := na.locf(na.locf(Filtered_CombinedEx, fromLast=TRUE), fromLast=FALSE)]
  
  #-----------------------------------------------------------------
  # (C) Convert Timestamp & Compute Time Differences (for velocities)
  #-----------------------------------------------------------------
  # Example: if your Timestamp is in format "HH:MM:SS" or "HH:MM:OS"
  # You may need a custom parse if it's something like "13:58.0".
  # Here is a simple approach that tries to convert "13:58.0" -> "13:58:00"
  
  if ("Timestamp" %in% colnames(df)) {
    df[, Timestamp_corrected := gsub("(\\d+:\\d+)\\.(\\d+)", "\\1:\\2", Timestamp)]
    df[, Timestamp := as.POSIXct(Timestamp_corrected, format="%H:%M:%S", tz="UTC")]
  }
  
  # Drop rows where Timestamp is NA
  df <- df[!is.na(Timestamp)]
  
  # Time difference (in seconds)
  df <- df %>%
    arrange(Timestamp) %>%
    mutate(Time_Diff = as.numeric(Timestamp - lag(Timestamp), units = "secs"))
  
  # Replace 0 or NA with small nonzero if you want
  df$Time_Diff[is.na(df$Time_Diff) | df$Time_Diff == 0] <- 1e-6
  
  #-----------------------------------------------------------------
  # (D) Compute Velocities (angle difference / time difference)
  #-----------------------------------------------------------------
  df <- df %>%
    mutate(
      Velocity_LeftEx      = (LeftEx - lag(LeftEx)) / Time_Diff,
      Velocity_RightEx     = (RightEx - lag(RightEx)) / Time_Diff,
      Velocity_CombinedEx  = (CombinedEx - lag(CombinedEx)) / Time_Diff,
      FiltVel_LeftEx       = (Filtered_LeftEx - lag(Filtered_LeftEx)) / Time_Diff,
      FiltVel_RightEx      = (Filtered_RightEx - lag(Filtered_RightEx)) / Time_Diff,
      FiltVel_CombinedEx   = (Filtered_CombinedEx - lag(Filtered_CombinedEx)) / Time_Diff
    )
  
  #-----------------------------------------------------------------
  # (E) Optional: Normalize the filtered velocities if desired
  #-----------------------------------------------------------------
  normalize_to_neg1_1 <- function(x) {
    rng <- range(x, na.rm=TRUE)
    if ((rng[2] - rng[1]) < 1e-6) {
      return(rep(0, length(x)))
    } else {
      return(2 * ((x - rng[1]) / (rng[2] - rng[1])) - 1)
    }
  }
  df <- df %>%
    mutate(
      Norm_FiltVel_LeftEx     = normalize_to_neg1_1(FiltVel_LeftEx),
      Norm_FiltVel_RightEx    = normalize_to_neg1_1(FiltVel_RightEx),
      Norm_FiltVel_CombinedEx = normalize_to_neg1_1(FiltVel_CombinedEx)
    )
  
  #-----------------------------------------------------------------
  # (F) Write the updated file back out
  #-----------------------------------------------------------------
  fwrite(df, file_path, row.names=FALSE)
  message("File updated with angles + velocities: ", file_path)
}

###############################################################################
# 2. Optional Pupillometry Analysis
###############################################################################
# If you want to do pupil data analysis, PupillometryR requires certain columns:
#   - subject, trial, time, + a pupil column
#   - e.g., "df$PupilRight", "df$PupilLeft"
#   - you can create a "mean_pupil" or keep them separate

# Example: process each CSV again to do a quick PupillometryR step
for (file_path in files) {
  
  message("Pupillometry: ", file_path)
  df <- fread(file_path)
  
  if (!all(c("LeftPupilDiameter","RightPupilDiameter","Timestamp") %in% colnames(df))) {
    message("No pupil columns or no Timestamp; skipping PupillometryR for ", file_path)
    next
  }
  
  # Minimal example: create a 'mean_pupil' column
  df$mean_pupil <- (df$LeftPupilDiameter + df$RightPupilDiameter)/2
  
  # For demonstration, define "Subject" and "Trial" if you have them; otherwise create dummy
  if (!("Subject" %in% colnames(df))) df$Subject <- "S1"
  if (!("Trial" %in% colnames(df)))   df$Trial   <- 1:nrow(df)
  
  # Make PupillometryR object
  pup_data <- make_pupillometryr_data(
    data      = df,
    subject   = Subject,
    trial     = Trial,
    time      = Timestamp,  # must be numeric or POSIXct
    condition = NULL         # or a column if you have conditions
  )
  
  # Run a quick median filter for demonstration
  # PupillometryR has many more features (regression, interpolation, baseline, etc.)
  filtered_data <- filter_data(data = pup_data,
                               pupil = mean_pupil,
                               filter = "median")
  
  # Save result as a separate file
  out_pupil <- sub("\\.csv$", "_pupillometry.csv", file_path)
  fwrite(as.data.table(filtered_data), out_pupil, row.names=FALSE)
  message("  Pupillometry data saved: ", out_pupil)
}

###############################################################################
# 3. Optional Saccade Detection
###############################################################################
# If you have X, Y screen coordinates and a time column, you can detect fixations
# and saccades with the saccades package. For example, using `LeftGazeX`/`LeftGazeY`.

for (file_path in files) {
  
  message("Saccade detection: ", file_path)
  df <- fread(file_path)
  
  # Check if we have 2D gaze columns
  if (!all(c("LeftGazeX","LeftGazeY","Timestamp") %in% colnames(df))) {
    message("No 2D gaze or Timestamp columns; skipping saccade detection for ", file_path)
    next
  }
  
  # Convert Timestamp to numeric or ms from start
  # If it's POSIXct, do something like:
  if (inherits(df$Timestamp, "character")) {
    # Attempt parse
    df$Timestamp <- as.POSIXct(df$Timestamp, format="%Y-%m-%d %H:%M:%S", tz="UTC")
  }
  if (inherits(df$Timestamp, "POSIXct")) {
    df$time_ms <- as.numeric(df$Timestamp - min(df$Timestamp), units="secs") * 1000
  } else {
    # If it's already numeric, rename:
    df$time_ms <- df$Timestamp
  }
  
  # saccades::detect.fixations expects a data frame with columns time, x, y
  sacc_df <- data.frame(
    time = df$time_ms,
    x    = df$LeftGazeX,
    y    = df$LeftGazeY
  )
  
  # If your data is noisy, try smoothing
  fixations <- detect.fixations(sacc_df,
                                smooth.coordinates = TRUE,
                                smooth.saccades    = FALSE)
  
  # Write saccade/fixation info
  out_sacc_file <- sub("\\.csv$", "_saccades.csv", file_path)
  write.csv(fixations, out_sacc_file, row.names=FALSE)
  message("  Saccades analysis written to: ", out_sacc_file)
}

message("All processing complete.")


