#  Store student IDs in a tuple
student_ids = ('S101', 'S102', 'S103', 'S104')

# Store student details in a dictionary
students = {
    'S101': {'name': 'Asha', 'assignment': 78, 'test': 80, 'attendance': 92, 'hours': 8},
    'S102': {'name': 'Ravi', 'assignment': 65, 'test': 68, 'attendance': 85, 'hours': 5},
    'S103': {'name': 'Meena', 'assignment': 88, 'test': 90, 'attendance': 96, 'hours': 10},
    'S104': {'name': 'Kiran', 'assignment': 55, 'test': 58, 'attendance': 78, 'hours': 4}
}

# Function to calculate average score
def calculate_average(assignment, test):
    return (assignment + test) / 2

# Function to determine academic risk level
def determine_risk(avg_score, attendance, hours):
    if avg_score < 60 or attendance < 80 or hours < 5:
        return "High Risk"
    elif avg_score < 75:
        return "Medium Risk"
    else:
        return "Low Risk"

# Process records and display performance report
print("STUDENT PERFORMANCE REPORT")
print("-" * 40)

for sid in student_ids:
    student = students[sid]
    
    avg = calculate_average(student['assignment'], student['test'])
    risk = determine_risk(avg, student['attendance'], student['hours'])
    
    print(f"Student ID   : {sid}")
    print(f"Name         : {student['name']}")
    print(f"Average Score: {avg:.2f}")
    print(f"Attendance   : {student['attendance']}%")
    print(f"Study Hours  : {student['hours']} hrs/week")
    print(f"Risk Level   : {risk}")
    print("-" * 40)
