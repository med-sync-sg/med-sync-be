from jinja2 import Environment, FileSystemLoader
import datetime
import json
import os

DEFAULT_PATH = os.path.dirname(os.path.abspath(__file__))
print(DEFAULT_PATH)

def generate_doctor_report(data: dict, is_doctor_report: bool):
    templates_path = os.path.join(DEFAULT_PATH, 'report_templates')
    env = Environment(loader=FileSystemLoader(templates_path))
    print(templates_path)
    
    if is_doctor_report:
        template = env.get_template('default_doctor_report.html')

        rendered_report = template.render(data)

        with open("doctor_report.html", "w", encoding="utf-8") as f:
            f.write(rendered_report)

        print("Report generated successfully! Check doctor_report.html")
        f.close()
    else:
        template = env.get_template('default_patient_report.html')

        rendered_report = template.render(data)
        with open("patient_report.html", "w", encoding="utf-8") as f:
            f.write(rendered_report)

            print("Report generated successfully! Check doctor_report.html")
        f.close()



