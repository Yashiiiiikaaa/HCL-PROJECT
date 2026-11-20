# PROJECT 23: ONLINE SHOPPING DELIVERY & LOGISTICS ANALYSIS

# Importing required libraries
import pandas as pd
import numpy as np


# --------------------------------------------------
# TASK 1: LOAD DATASET AND INSPECT DATA
# --------------------------------------------------

# Read CSV file
df = pd.read_csv("23_delivery_performance_extended.csv")

print("\n----- DATASET INFORMATION -----")

# Display first five rows
print(df.head())

# Find number of rows and columns
print("Dataset Shape:", df.shape)

# Display column names
print("Columns:", df.columns.tolist())

# Display data types
print("Data Types:")
print(df.dtypes)


# --------------------------------------------------
# TASK 2: FIND MISSING VALUES AND DUPLICATES
# --------------------------------------------------

print("\n----- MISSING VALUES -----")

# Count missing values in each column
print(df.isnull().sum())

# Count duplicate Shipment_ID records
print("Duplicate Shipment IDs:",
      df.duplicated(subset="Shipment_ID").sum())


# --------------------------------------------------
# TASK 3: HANDLE MISSING VALUES
# --------------------------------------------------

# Remove duplicate shipment records
df = df.drop_duplicates(subset="Shipment_ID", keep="first")

# Fill missing Shipping_Cost using median
median_cost = df["Shipping_Cost"].median()

df["Shipping_Cost"] = df["Shipping_Cost"].fillna(median_cost)

# Missing Delivery_Date values are kept as NaT.
# We should not invent delivery dates for shipments.

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# --------------------------------------------------
# TASK 4: CONVERT DATE COLUMNS INTO DATETIME
# --------------------------------------------------

df["Order_Date"] = pd.to_datetime(
    df["Order_Date"], errors="coerce"
)

df["Delivery_Date"] = pd.to_datetime(
    df["Delivery_Date"], errors="coerce"
)

print("\nDate columns converted successfully.")


# --------------------------------------------------
# TASK 5: CALCULATE DELIVERY DAYS
# --------------------------------------------------

df["Delivery_Days"] = (
    df["Delivery_Date"] - df["Order_Date"]
).dt.days

print("\nDelivery Days:")
print(df[["Shipment_ID", "Delivery_Days"]].head())


# --------------------------------------------------
# TASK 6: TOTAL SHIPMENTS AND SUCCESS PERCENTAGE
# --------------------------------------------------

total_shipments = len(df)

delivered_shipments = (
    df["Order_Status"] == "Delivered"
).sum()

success_percentage = (
    delivered_shipments / total_shipments
) * 100

print("\n----- DELIVERY SUMMARY -----")

print("Total Shipments:", total_shipments)

print("Delivered Shipments:", delivered_shipments)

print("Delivery Success Percentage:",
      round(success_percentage, 2), "%")


# --------------------------------------------------
# TASK 7: AVERAGE DELIVERY DAYS BY SHIPPING METHOD
# --------------------------------------------------

average_delivery = df.groupby(
    "Shipping_Method"
)["Delivery_Days"].mean()

print("\n----- AVERAGE DELIVERY BY METHOD -----")
print(average_delivery)


# --------------------------------------------------
# TASK 8: RANK DELIVERY PARTNERS
# --------------------------------------------------

partner_ranking = df.groupby(
    "Delivery_Partner"
).agg(
    Total_Shipments=("Shipment_ID", "count"),
    Average_Days=("Delivery_Days", "mean")
)

# Rank partners by number of shipments
partner_ranking = partner_ranking.sort_values(
    by="Total_Shipments", ascending=False
)

print("\n----- DELIVERY PARTNER RANKING -----")
print(partner_ranking)


# --------------------------------------------------
# TASK 9: SUCCESS RATE BY SHIPPING METHOD
# --------------------------------------------------

# Create a new column for delivered status
df["Is_Delivered"] = (
    df["Order_Status"] == "Delivered"
)

method_success = df.groupby(
    "Shipping_Method"
)["Is_Delivered"].mean() * 100

print("\n----- SUCCESS RATE BY SHIPPING METHOD -----")
print(method_success.round(2))


# --------------------------------------------------
# TASK 10: TOP 10 LONGEST DELIVERIES
# --------------------------------------------------

longest_deliveries = df.sort_values(
    by="Delivery_Days", ascending=False
).head(10)

print("\n----- TOP 10 LONGEST DELIVERIES -----")

print(longest_deliveries[
    ["Shipment_ID", "Delivery_Partner", "Delivery_Days"]
])


# --------------------------------------------------
# TASK 11: DICTIONARY AND SET
# --------------------------------------------------

# Dictionary to store shipment counts by status
status_counts = df["Order_Status"].value_counts().to_dict()

print("\n----- SHIPMENT COUNTS BY STATUS -----")
print(status_counts)

# Set to store unique delivery partners
unique_partners = set(df["Delivery_Partner"].dropna())

print("\nUnique Delivery Partners:")
print(unique_partners)


# --------------------------------------------------
# TASK 12: FUNCTION TO CLASSIFY DELIVERY DELAYS
# --------------------------------------------------

# Function to classify delivery duration

def classify_delivery(days):

    if pd.isna(days):
        return "Not Delivered / Unknown"

    elif days <= 3:
        return "On-Time"

    elif days <= 7:
        return "Slightly Delayed"

    else:
        return "Highly Delayed"


# Apply function to every shipment
df["Delivery_Category"] = df["Delivery_Days"].apply(
    classify_delivery
)

print("\n----- DELIVERY CATEGORIES -----")

print(df[
    ["Shipment_ID", "Delivery_Days", "Delivery_Category"]
].head(10))


# --------------------------------------------------
# TASK 13: NUMPY STATISTICAL CALCULATIONS
# --------------------------------------------------

# Remove missing values before calculations
delivery_values = df["Delivery_Days"].dropna().to_numpy()

# Calculate mean
mean_days = np.mean(delivery_values)

# Calculate median
median_days = np.median(delivery_values)

# Calculate standard deviation
std_days = np.std(delivery_values)

print("\n----- NUMPY STATISTICS -----")

print("Mean:", round(mean_days, 2))

print("Median:", round(median_days, 2))

print("Standard Deviation:", round(std_days, 2))


# --------------------------------------------------
# TASK 14: IQR METHOD AND MONTHLY ANALYSIS
# --------------------------------------------------

# Calculate Q1 and Q3
Q1 = np.percentile(delivery_values, 25)

Q3 = np.percentile(delivery_values, 75)

# Calculate IQR
IQR = Q3 - Q1

# Calculate upper and lower limits
lower_limit = Q1 - 1.5 * IQR

upper_limit = Q3 + 1.5 * IQR

print("\n----- IQR ANALYSIS -----")

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)

# Find unusually long deliveries
outliers = df[
    df["Delivery_Days"] > upper_limit
]

print("\nUnusually Long Deliveries:")
print(outliers[
    ["Shipment_ID", "Delivery_Days"]
])

# Monthly delivery performance
df["Month"] = df["Order_Date"].dt.month_name()

monthly_analysis = df.groupby("Month")[
    "Delivery_Days"
].mean()

print("\n----- MONTHLY DELIVERY PERFORMANCE -----")
print(monthly_analysis)


# --------------------------------------------------
# TASK 15: OOPS - CLASS AND OBJECT
# --------------------------------------------------

# Creating a Shipment class

class Shipment:

    # Constructor
    def __init__(self, shipment_id, partner, delivery_days):

        self.shipment_id = shipment_id
        self.partner = partner
        self.delivery_days = delivery_days

    # Method to display shipment details
    def display_details(self):

        print("\nShipment ID:", self.shipment_id)

        print("Delivery Partner:", self.partner)

        print("Delivery Days:", self.delivery_days)


# Creating an object of Shipment class

shipment1 = Shipment("SH1001", "ParcelPro", 7)

# Calling class method
shipment1.display_details()


# --------------------------------------------------
# FIVE RECOMMENDATIONS
# --------------------------------------------------

recommendations = [

    "Improve delivery partner performance monitoring.",

    "Use faster shipping methods for urgent orders.",

    "Track shipments regularly to identify delays.",

    "Improve warehouse and inventory management.",

    "Provide customers with accurate delivery updates."

]

print("\n----- FIVE RECOMMENDATIONS -----")

# Display recommendations using a loop
for i in range(len(recommendations)):

    print(i + 1, recommendations[i])

# --------------------------------------------------
# TASK 16: FILE HANDLING
# --------------------------------------------------

# Writing project summary into a text file

with open("delivery_project_report.txt", "w") as file:

    file.write("ONLINE SHOPPING DELIVERY ANALYSIS\n")

    file.write("---------------------------------\n")

    file.write("Total Shipments: " + str(total_shipments) + "\n")

    file.write("Delivered Shipments: " + str(delivered_shipments) + "\n")

    file.write("Success Percentage: " + str(round(success_percentage, 2)) + "%\n")

    file.write("Average Delivery Days: " + str(round(mean_days, 2)) + "\n")

    file.write("Unique Delivery Partners: " + str(len(unique_partners)) + "\n")


# Appending recommendations to the same file

with open("delivery_project_report.txt", "a") as file:

    file.write("\nFIVE RECOMMENDATIONS:\n")

    for i in range(len(recommendations)):

        file.write(str(i + 1) + ". " + recommendations[i] + "\n")


# Reading the saved report

with open("delivery_project_report.txt", "r") as file:

    print("\n----- PROJECT REPORT FROM FILE -----")

    print(file.read())

print("Report generated successfully!")


# --------------------------------------------------
# FINAL PROJECT SUMMARY
# --------------------------------------------------

print("\n----- PROJECT COMPLETED SUCCESSFULLY -----")

print("Total Shipments:", total_shipments)

print("Delivered Shipments:", delivered_shipments)

print("Average Delivery Days:", round(mean_days, 2))

print("Unique Delivery Partners:", len(unique_partners))