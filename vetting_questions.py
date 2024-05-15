from langchain.document_loaders.csv_loader import CSVLoader
import re
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List


# loader = CSVLoader(file_path="_Software Vetting Request and Resolution List - SW Vetting Template.csv")
# data = loader.load()

# extracted_questions = [row.page_content.split('\n: ')[1].split('\n')[0] for row in data[31:74]]

# extracted_fields = []
# for string in extracted_questions:
#     match = re.search(r'(\d+\.\d+)\s+(.*)', string)
#     if match:
#         numbered_field = match.group(1)
#         text = match.group(2)
#         extracted_fields.append((numbered_field, text))
#
# extracted_dict_list = [{'number': field, 'question': text} for field, text in extracted_fields]


extracted_questions= [
    "",  # Blank option
    "Does the App Provider disclose the presence and use of third party cookies and provide options for managing them?",
    "Does the App Provider use User data for statistical analysis and profiling, for making subjective assessments, for predicting behaviour or as part of a decision-making process?",
    "If the App Provider uses User data for statistical analysis and profiling, for making subjective assessments, for predicting behaviour or as part of a decision-making process, is it clearly communicated to Users?",
    "If the App Provider uses User data for statistical analysis and profiling, for making subjective assessments, for predicting behaviour or as part of a decision-making process, is there a mechanism for Users to challenge these assessments?",
    "Does the App Provider enable Users to erase their data, including metadata inferences, assessments and profiles (if not required for administrative purposes by the provider or the school board)?",
    "Does the App Provider enable Users to erase their data, without any fee or change for this service?",
    "Does the App provider ensure that when a User deletes their work in their account created by an administrator where the administrator maintains exclusive administrative rights, the copies in the administrator account also disappear?",
    "Does the App Provider provide contact information of an operator who will respond to inquiries and challenges from Users about privacy policies, data handling practices, accuracy, completeness and use of personal information?",
    "Does the App Provider provide Users a mechanism to access, correct, erase, and download content Users create, in a useable format?",
    "Do Users of the App have the ability to delete accounts associated with the App, web app, software and/or service?",
    "Does the App provider require the User to surrender copyright to their work if they post it to the Application or service’s site?",
    "Does the App Provider state whether or not the business app/service allows users to make personal information publicly available online?",
    "Does the App Provider communicate privacy notices, permissions, privacy policies, Terms of Service, contracts etc., in clear, specific and unambiguous language?",
    "Does the App Provider communicate to Users how their personal information is being used, processed, disclosed and retained by the provider and any third parties?",
    "Does the App Provider make links to permissions, privacy policies and Terms of Service, etc., easy to find after the account has been created?",
    "Does the App Provider identify the third parties to which they disclose personal information for processing?",
    "Does the App Provider identify the specific personal information data elements they share with third parties?",
    "Does the App Provider provide a summary of protections/assurances in place for personal information shared with third parties?",
    "Does the App Provider directly inform Users before changes are made to policies and terms of use, etc., before data is used in a manner inconsistent with the terms they were initially provided?",
    "Does the App Provider confirm that they comply with all laws in their jurisdiction?"
]

modifier_terms = [
    "Extremely Brief",
    "Minimalist",
    "Brief",
    "Concise",
    "Compact",
    "Focused",
    "In-depth",
    "Ultra-detailed",
    "All-Encompassing",
    "Unabridged and Expansive",
    "Painstakingly Detailed"
]

class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)
