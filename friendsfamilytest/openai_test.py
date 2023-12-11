from openai import OpenAI
client = OpenAI()

input_list = [
    "more face to face appointments",
    "shorter appointment wait times",
    "Drinking fountain in reception would be great if you are thursty or need to take any medication",
    "Flu jabs more readily available",
    "I can never get an appointment when I need one on a Monday",
    "It is difficult to get hold of the Pharmacist when I meed a repeat prescription",
    "no everything was good",
    "thanks",
    "I am very happy with the service"
]

system_prompt = """you are an expert practice manager for a GP Surgery, you will review improvement suggestions from patients and classify them into one of the following categories:
improvement_labels_list = [
        "Reception Services",
        "Ambiance of Facility",
        "Facility Modernisation and Upgrades",
        "Nursing Quality",
        "Waiting Times",
        "Referral Process",
        "Staffing Levels",
        "Facility Accessibility",
        "Poor Communication",
        "Online Services & Digital Health",
        "Patient Safety",
        "Weekend Service Availability",
        "Telephone Service",
        "After-Hours Service",
        "Staff Training and Development",
        "Prescription Process",
        "Quality of Medical Advice",
        "Overall Patient Satisfaction",
        "Appointment System Efficiency",
        "Blood Test Results & Imaging",
        "Patient Participation Group",
        "Mental Health Services",
        "Social Prescribing Services",
        "Chronic Disease Management",
        "No Improvement Suggestion",
        "Doctor Consultations",
        "Home Visits",
        "Cancer Screening",
        "Vaccinations",
        "Test Results",
        "Clinical Pharmacist",
    ]
when you respond only select the most appropriate category and only return the category as specified in the 'improvement_labels_list', do not return any other text. if the text provided does not fit into an improvement suggestion category classify it as 'No Improvement Suggestion'
thank you"""


for input in input_list:
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input}
    ]
    )

    print(completion.choices[0].message.content)
