import re

#Student Class 
class Student:
    def __init__(self, student_id, name):
        self.student_id = student_id
        self.name = name
        self.activities = []

    def add_activity(self, activity, date, time):
        self.activities.append((activity, date, time))

    def summary(self):
        logins = 0
        submissions = 0
        login_stack = 0
        abnormal = False
        daily_stats = {}

        for act, date, time in self.activities:
            daily_stats[date] = daily_stats.get(date, 0) + 1

            if act == "LOGIN":
                logins += 1
                login_stack += 1
            elif act == "LOGOUT":
                login_stack = max(0, login_stack - 1)
            elif act == "SUBMIT_ASSIGNMENT":
                submissions += 1

        if login_stack > 0:
            abnormal = True

        return logins, submissions, abnormal, daily_stats


# Generator to Read Log File
def read_log_file(filename):
    pattern = re.compile(
        r"^(S\d+)\s*\|\s*(\w+)\s*\|\s*(LOGIN|LOGOUT|SUBMIT_ASSIGNMENT)\s*\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(\d{2}:\d{2})$"
    )

    with open(filename, "r") as file:
        for line in file:
            line = line.strip()

            # Ignore empty lines and comments
            if not line or line.startswith("#"):
                continue

            try:
                match = pattern.match(line)
                if match:
                    yield match.groups()
                else:
                    raise ValueError("Invalid log entry")
            except Exception:
                print("Invalid entry skipped:", line)



# Main Program
students = {}

input_file = "student_logs.txt"
output_file = "activity_report.txt"

for sid, name, activity, date, time in read_log_file(input_file):
    if sid not in students:
        students[sid] = Student(sid, name)
    students[sid].add_activity(activity, date, time)

#Generate Report 
with open(output_file, "w") as out:
    print("\nSTUDENT ACTIVITY REPORT\n")
    out.write("STUDENT ACTIVITY REPORT\n\n")

    for student in students.values():
        logins, submits, abnormal, daily = student.summary()

        report = (
            f"Student ID: {student.student_id}\n"
            f"Name: {student.name}\n"
            f"Total Logins: {logins}\n"
            f"Total Submissions: {submits}\n"
            f"Abnormal Behavior: {'YES' if abnormal else 'NO'}\n"
            f"Daily Activity Stats: {daily}\n"
            f"{'-'*40}\n"
        )

        print(report)
        out.write(report)

