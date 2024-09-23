import numpy as np
import datetime as dt


# Checkpoint function
def checkpoint(file_name, checkpoint_header, checkpoint_data):
    np.savez(file_name, header=checkpoint_header, data=checkpoint_data)
    checkpoint_variable = np.load(file_name + ".npz")
    return checkpoint_variable


# Import data
np.set_printoptions(suppress=True, linewidth=100, precision=2)
absenteeism_raw_data = np.genfromtxt("Absenteeism-data.csv",
                                     delimiter=",",
                                     skip_header=1,
                                     autostrip=True,
                                     )
print(absenteeism_raw_data)

# Import column headers
absenteeism_headers = np.genfromtxt("Absenteeism-data.csv",
                                            delimiter=",",
                                            skip_footer=absenteeism_raw_data.shape[0],
                                            autostrip=True,
                                            dtype=str
                                            )
print(absenteeism_headers)

# Check for missing data
print(np.isnan(absenteeism_raw_data).sum())

# Temporary max value to replace missing values
temporary_fill_max = np.nanmax(absenteeism_raw_data) + 1
print(temporary_fill_max)

# Temporary mean value to  heck which columns contain nan values
temporary_mean = np.nanmean(absenteeism_raw_data, axis=0)
print(temporary_mean)

# To be used later to fill in values
temporary_stats = np.array([np.nanmin(absenteeism_raw_data, axis=0),
                            np.nanmean(absenteeism_raw_data, axis=0),
                            np.nanmax(absenteeism_raw_data, axis=0)])
print(temporary_stats)

# Splitting dataset
column_numeric = np.argwhere(np.isnan(temporary_mean) == False).squeeze()
print(column_numeric)

column_date = np.argwhere(np.isnan(temporary_mean)).squeeze()
print(column_date)

# Import column headers
absenteeism_numeric_headers = np.genfromtxt("Absenteeism-data.csv",
                                            delimiter=",",
                                            skip_footer=absenteeism_raw_data.shape[0],
                                            autostrip=True,
                                            usecols=column_numeric,
                                            dtype=str
                                            )
absenteeism_headers_v1 = absenteeism_numeric_headers  # For my personal reference
print(absenteeism_numeric_headers)

# Creating dataset for both types of data and replacing missing values in the numeric dataset with the max value
absenteeism_numeric_data = np.genfromtxt("Absenteeism-data.csv",
                                         delimiter=",",
                                         skip_header=1,
                                         autostrip=True,
                                         usecols=column_numeric,
                                         filling_values=temporary_fill_max,
                                         dtype=int
                                         )
print(absenteeism_numeric_data)
absenteeism_data_date = np.genfromtxt("Absenteeism-data.csv",
                                      delimiter=",",
                                      skip_header=1,
                                      autostrip=True,
                                      usecols=column_date,
                                      dtype=str
                                      )
print(absenteeism_data_date)

# Dropping the "ID" column
absenteeism_numeric_data = np.delete(absenteeism_numeric_data, 0, axis=1)
absenteeism_numeric_headers = np.delete(absenteeism_numeric_headers, 0)
print(absenteeism_numeric_headers)
print(absenteeism_numeric_data)


# Checking values in the second column "Reason for Absence"
print(absenteeism_headers_v1)
print(np.unique(absenteeism_raw_data[:, 1]))

# Spliting the columns into dummy variables with 4 different groupings
group_1 = np.arange(1, 15)
group_2 = np.arange(15, 18)
group_3 = np.arange(18, 22)
group_4 = np.arange(22, 29)

reason_1 = np.where(np.isin(absenteeism_raw_data[:, 1], group_1) == True, 1, 0)
reason_2 = np.where(np.isin(absenteeism_raw_data[:, 1], group_2) == True, 1, 0)
reason_3 = np.where(np.isin(absenteeism_raw_data[:, 1], group_3) == True, 1, 0)
reason_4 = np.where(np.isin(absenteeism_raw_data[:, 1], group_4) == True, 1, 0)

reason_1 = np.reshape(reason_1, (700, 1))
reason_2 = np.reshape(reason_2, (700, 1))
reason_3 = np.reshape(reason_3, (700, 1))
reason_4 = np.reshape(reason_4, (700, 1))

# Deleting the "Reason for Absence" column and adding the new columns with dummy variables
absenteeism_numeric_data = np.delete(absenteeism_numeric_data, 0, axis=1)
absenteeism_numeric_headers = np.delete(absenteeism_numeric_headers, 0)
absenteeism_numeric_data = np.hstack((reason_1, reason_2, reason_3, reason_4, absenteeism_numeric_data))
reason_column_headers = np.array(["Reason_1", "Reason_2", "Reason_3", "Reason_4"])
absenteeism_numeric_headers = np.concatenate((reason_column_headers, absenteeism_numeric_headers))
print(absenteeism_numeric_data)
print(absenteeism_numeric_headers)

# Converting data in the education into binary data
print(absenteeism_numeric_headers)
print(np.unique(absenteeism_numeric_data[:, 9]))
absenteeism_numeric_data[:, 9] = np.array([0 if numbers == 1 else 1
                                           for numbers in absenteeism_numeric_data[:, 9]])
print(absenteeism_numeric_data[:, 9])

# Checkpoint 1
processed_numeric_data = checkpoint("Checkpoint-Numeric", absenteeism_numeric_headers, absenteeism_numeric_data)

# Extracting the month value from the "Date" column
print(absenteeism_data_date)
absenteeism_data_month = np.array([date[4] if date[3] == "0" else date[3:5] for date in absenteeism_data_date])
print(absenteeism_data_month)


# Extracting the day of the week value from the "Date" column
print(absenteeism_data_date)
absenteeism_data_day_of_the_week = np.array([dt.datetime(int(date[6:10]), int(date[3:5]), int(date[0:2])).weekday() + 1
                                             for date in absenteeism_data_date], dtype=str)
print(absenteeism_data_day_of_the_week)

# Formatting new Date columns to add to the numeric dataset
absenteeism_data_month = np.reshape(absenteeism_data_month, (700, 1))
absenteeism_data_day_of_the_week = np.reshape(absenteeism_data_day_of_the_week, (700, 1))
final_absenteeism_data_date = np.hstack((absenteeism_data_month, absenteeism_data_day_of_the_week))

# Column headers for new date columns
final_date_headers = np.array(["Month", "Day of the Week"])

# Checkpoint 2
processed_date_data = checkpoint("Checkpoint-Dates", final_date_headers, final_absenteeism_data_date)

# Combining headers with their respective data
final_numeric_data = np.vstack((processed_numeric_data["header"], processed_numeric_data["data"]))
final_date_data = np.vstack((processed_date_data["header"], processed_date_data["data"]))

# Combining numeric data with date data
final_absenteeism_data = np.hstack((final_numeric_data, final_date_data))

# Storing the new Dataset
np.savetxt("absenteeism-data-preprocessed.csv",
           final_absenteeism_data,
           fmt="%s",
           delimiter=",")





