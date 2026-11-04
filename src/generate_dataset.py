"""
generate_dataset.py
-------------------
Creates data/tickets.csv  (the raw dataset).

The ticket descriptions below were written by hand (30 per category, 5 categories).
Priority and status are assigned randomly (with a fixed seed so results are repeatable).

To make Task 1 (data cleaning) meaningful, we deliberately add:
  - a few duplicate rows
  - a few missing values
Run once:  python src/generate_dataset.py
"""

import random
from pathlib import Path

import pandas as pd

random.seed(42)  # same random numbers every run -> same dataset every run

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "data" / "tickets.csv"

TICKETS = {
    "Login Issue": [
        "I forgot my password and cannot login",
        "Password reset email is not arriving in my inbox",
        "My account is locked after too many failed login attempts",
        "Unable to sign in to the portal with my username",
        "The reset password link says it has expired",
        "I keep getting invalid credentials even though my password is correct",
        "Cannot log in to the mobile app",
        "Login page keeps redirecting me back to the sign in screen",
        "I did not receive the OTP to log into my account",
        "Two factor authentication code is not working",
        "My login is not working after I changed my password",
        "Please unlock my account, I am unable to access the system",
        "Sign in button does nothing when I enter my credentials",
        "I am locked out of my account since this morning",
        "Password reset is not working",
        "Cannot access my account because the verification code never comes",
        "Forgot my username and need help to log in",
        "Single sign on is failing for my user",
        "It says my account is disabled when I try to login",
        "Unable to authenticate with my email and password",
        "The system rejects my password every time I try to sign in",
        "I need help recovering my login details",
        "I get an authentication failed message on the login screen",
        "My new password is not being accepted",
        "Login keeps failing on Chrome browser",
        "I cannot get past the login page",
        "Security question for password recovery is not accepting my answer",
        "My account got logged out and now I cannot sign in again",
        "Unable to login using my company email id",
        "The login screen shows incorrect username or password",
    ],
    "Application Error": [
        "The application is showing an error while saving data",
        "I get a 500 internal server error when I submit the form",
        "The app crashes whenever I click on the dashboard",
        "Error message appears when I try to upload a file",
        "Application gives error while saving",
        "The page shows something went wrong and nothing loads",
        "I am getting a database connection error in the application",
        "The submit button throws an unexpected exception",
        "App closes suddenly when I open the settings page",
        "Getting error code 404 when opening the invoice page",
        "The application displays a blank white screen with an error",
        "Data is not getting saved and an error pops up",
        "An unknown error occurred while creating a new record",
        "The export button gives an error message",
        "Application crashed and I lost my unsaved work",
        "I see a null pointer error on the order screen",
        "The system throws an error when I try to delete an entry",
        "Error while attaching a document to the ticket",
        "The software freezes and then shows a fatal error",
        "Getting a timeout error message while submitting the form",
        "The checkout page fails with a server error",
        "Bug in the application: clicking save causes an error popup",
        "The application stopped working after the latest update",
        "Error occurs when I try to edit my saved record",
        "I get a failed to load data error on the customer page",
        "The app displays an exception message on startup",
        "Payment page shows error and does not process the request",
        "Unable to update the record because the app returns an error",
        "Application error occurred while sending email notification",
        "The form crashes with an error whenever I add an attachment",
    ],
    "Report": [
        "Please help me generate my monthly sales report",
        "Need monthly sales report",
        "I need a report of all tickets closed last week",
        "Can you send me the quarterly revenue report",
        "How do I download the attendance report",
        "The report is showing wrong numbers for last month",
        "I want to schedule a weekly report to be emailed to me",
        "Need an annual performance summary report for my team",
        "Please share the inventory report for this quarter",
        "I need to save the report as a PDF file",
        "I need a custom report filtered by region",
        "Request for a customer growth report for the last year",
        "Where can I find the daily sales summary report",
        "The report totals do not match the dashboard figures",
        "Please generate an expense report for October",
        "I would like a report showing the number of orders per product",
        "Can you add a new column to the monthly report",
        "Need the year end financial report",
        "Request a report on employee leave balances",
        "Please provide the list of pending invoices as a report",
        "Generate a report of sales by customer for this quarter",
        "I want to see the trend report for the past six months",
        "Could you help me create a new report template",
        "Need a report of all open tickets by department",
        "Please send me the stock level report",
        "How can I print the monthly summary report",
        "Requesting the profit and loss report for last quarter",
        "The scheduled report did not arrive this week",
        "I need a data report for the audit team",
        "Kindly generate the weekly progress report",
    ],
    "Account Update": [
        "I need to change my registered mobile number",
        "Please update my email address on my account",
        "I want to change the name on my profile",
        "Need to update my billing address",
        "Please change my registered phone number",
        "How can I update my date of birth in my account",
        "I moved to a new city and need to update my address",
        "Kindly update my company name in the profile",
        "I want to add a secondary email to my account",
        "Please remove my old phone number and add the new one",
        "My name is spelled wrong in my profile, please correct it",
        "I want to change my account username",
        "Need to update my bank details for refunds",
        "Please update my contact details",
        "I would like to change my notification preferences",
        "Update my shipping address for future orders",
        "How do I change the email linked to my account",
        "I want to upgrade my subscription plan",
        "Please close my old profile and update my account information",
        "Need to change the primary contact person on the account",
        "I got a new number and want it registered with my account",
        "Please change the language setting on my profile",
        "Edit my profile picture and job title",
        "I need to correct my tax id on the account",
        "Can you change the account holder name",
        "Please modify my registered email id",
        "I want to update my communication preferences",
        "Need to add a new phone number to my profile",
        "Change my designation in the account details",
        "Please update my address in the customer record",
    ],
    "Performance": [
        "The application is very slow today",
        "Application is very slow",
        "The pages take too long to load",
        "System is lagging when I open large files",
        "The dashboard takes more than a minute to load",
        "Everything is running extremely slow since morning",
        "The app is taking too long to respond",
        "Search results are taking forever to show up",
        "Website is very sluggish during peak hours",
        "The application is taking too long to load",
        "Response time of the system has become really bad",
        "Screen keeps lagging while I scroll through records",
        "Opening the customer list is painfully slow",
        "The app has become much slower after the last update",
        "Home page loads very slowly",
        "Saving records takes several minutes",
        "The portal is not responsive and keeps hanging",
        "System performance is poor and delays my work",
        "It takes ages to switch between tabs in the application",
        "Video calls in the tool are laggy and slow",
        "High latency while using the application today",
        "Upload is extremely slow on the platform",
        "The system is slow when many users are online",
        "Page loading time has increased a lot this week",
        "The app is sluggish and unresponsive",
        "Menu options take a long time to open",
        "Application speed is too slow to do any work",
        "Everything freezes for a few seconds after every click",
        "Our team is facing slow performance in the system all day",
        "The software is working very slowly on my laptop",
    ],
}

# Priority is slightly related to category (e.g. login problems are often urgent)
PRIORITY_WEIGHTS = {
    "Login Issue":       [0.60, 0.30, 0.10],   # High, Medium, Low
    "Application Error": [0.55, 0.35, 0.10],
    "Report":            [0.10, 0.50, 0.40],
    "Account Update":    [0.10, 0.40, 0.50],
    "Performance":       [0.45, 0.40, 0.15],
}
PRIORITIES = ["High", "Medium", "Low"]
STATUSES = ["Open", "Closed", "In Progress"]
STATUS_WEIGHTS = [0.40, 0.45, 0.15]


def build_dataset() -> pd.DataFrame:
    rows = []
    for category, descriptions in TICKETS.items():
        for text in descriptions:
            rows.append(
                {
                    "ticket_description": text,
                    "category": category,
                    "priority": random.choices(PRIORITIES, PRIORITY_WEIGHTS[category])[0],
                    "status": random.choices(STATUSES, STATUS_WEIGHTS)[0],
                }
            )

    random.shuffle(rows)  # mix categories so the file is not sorted
    df = pd.DataFrame(rows)
    df.insert(0, "ticket_id", [f"T{i:03d}" for i in range(1, len(df) + 1)])

    # --- Deliberately "dirty" the data so Task 1 has something to clean ---
    # 1) a few missing values
    df.loc[10, "priority"] = None
    df.loc[45, "priority"] = None
    df.loc[90, "status"] = None
    df.loc[120, "status"] = None
    df.loc[75, "ticket_description"] = None   # a ticket with no text

    # 2) a few exact duplicate rows
    duplicates = df.iloc[[3, 25, 60, 100]]
    df = pd.concat([df, duplicates], ignore_index=True)

    return df


if __name__ == "__main__":
    data = build_dataset()
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(data)} rows to {OUTPUT_PATH}")
