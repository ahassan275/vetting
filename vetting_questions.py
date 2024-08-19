# from langchain.document_loaders.csv_loader import CSVLoader
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


# extracted_questions= [
#     "",  # Blank option
#     "Does the App Provider disclose the presence and use of third party cookies and provide options for managing them?",
#     "Does the App Provider use User data for statistical analysis and profiling, for making subjective assessments, for predicting behaviour or as part of a decision-making process?",
#     "If the App Provider uses User data for statistical analysis and profiling, for making subjective assessments, for predicting behaviour or as part of a decision-making process, is it clearly communicated to Users?",
#     "If the App Provider uses User data for statistical analysis and profiling, for making subjective assessments, for predicting behaviour or as part of a decision-making process, is there a mechanism for Users to challenge these assessments?",
#     "Does the App Provider enable Users to erase their data, including metadata inferences, assessments and profiles (if not required for administrative purposes by the provider or the school board)?",
#     "Does the App Provider enable Users to erase their data, without any fee or change for this service?",
#     "Does the App provider ensure that when a User deletes their work in their account created by an administrator where the administrator maintains exclusive administrative rights, the copies in the administrator account also disappear?",
#     "Does the App Provider provide contact information of an operator who will respond to inquiries and challenges from Users about privacy policies, data handling practices, accuracy, completeness and use of personal information?",
#     "Does the App Provider provide Users a mechanism to access, correct, erase, and download content Users create, in a useable format?",
#     "Do Users of the App have the ability to delete accounts associated with the App, web app, software and/or service?",
#     "Does the App provider require the User to surrender copyright to their work if they post it to the Application or service’s site?",
#     "Does the App Provider state whether or not the business app/service allows users to make personal information publicly available online?",
#     "Does the App Provider communicate privacy notices, permissions, privacy policies, Terms of Service, contracts etc., in clear, specific and unambiguous language?",
#     "Does the App Provider communicate to Users how their personal information is being used, processed, disclosed and retained by the provider and any third parties?",
#     "Does the App Provider make links to permissions, privacy policies and Terms of Service, etc., easy to find after the account has been created?",
#     "Does the App Provider identify the third parties to which they disclose personal information for processing?",
#     "Does the App Provider identify the specific personal information data elements they share with third parties?",
#     "Does the App Provider provide a summary of protections/assurances in place for personal information shared with third parties?",
#     "Does the App Provider directly inform Users before changes are made to policies and terms of use, etc., before data is used in a manner inconsistent with the terms they were initially provided?",
#     "Does the App Provider confirm that they comply with all laws in their jurisdiction?"
# ]


extracted_questions = [
    "",  # Blank option
    "Does <Vendor/App> state all data elements that their business web apps or services collect?",
    "Does <Vendor/App> provide reasons for the collection/processing of each data elements that their business web apps or services collect?",
    "Does <Vendor/App> ensure there is verifiable User consent for the collection, use and disclosure of personal information (unless there is a legal requirement from law enforcement or regulators)?",
    "Does <Vendor/App> allow Users to control and maintain ownership of content they create and upload to the App?",
    "Does <Vendor/App> offer Users consent options for the collection and use of User personal information necessary to provide the service, without consenting to the use or disclosure of that information to third parties for other purposes (e.g. marketing)?",
    "Does <Vendor/App> collect only the personal information required to operate the business web app/software/service (e.g. no automatic access of browser history, contact lists, search terms, preferences, device identification, location, etc.), unless directly related to providing the service?",
    "When a User installs the <Vendor/App> App onto their mobile device, are Users offer choices regarding disclosure of data on their device (e.g. location, identifiers, contacts, etc.)?",
    "Does <Vendor/App> covertly collect personal information (i.e., without the User's knowledge, or from the User's own device)?",
    "Does <Vendor/App> keep User profiles and activity within the App private, so that third-parties cannot observe User activity, nor collect User information?",
    "Does <Vendor/App> allow Users to create generic accounts (e.g. role-based, guests, etc.)?",
    "Does <Vendor/App> allow Users to create profiles using as little personal information as possible (in order to avoid the excessive collection of personal information)?",
    "Does <Vendor/App> use, disclose and retain personal information only for the purpose of providing the business web app/software/service?",
    "Does <Vendor/App> benefit or profit from sharing User personal information or targeting users for commercial purposes?",
    "Does <Vendor/App> profile users for marketing purposes or in ways that could lead to unfair, unethical or discriminatory treatment?",
    "Does <Vendor/App> repurpose user data or use it for research without express User consent?",
    "Does <Vendor/App> anonymize User data if used for research or shared with third-parties?",
    "Does <Vendor/App> securely destroy or make anonymous in a timely manner all personal information that is no longer required to provide the app/software/service?",
    "Does <Vendor/App> explicitly state User data retention timelines?",
    "Does <Vendor/App> have a comprehensive security program in place that is reasonably designed to protect the security, privacy, confidentiality, and integrity of User personal information?",
    "What administrative, technological, and physical safeguards does <Vendor/App> use to protect against risks of User data being accessed and/or altered by unauthorized parties?",
    "Does <Vendor/App> ensure that all of its third-party partners implement the same security safeguards <Vendor/App> does?",
    "Does <Vendor/App> require its successor entities to implement <Vendor/App>'s security safeguards for User personal information?",
    "Does <Vendor/App> state what its breach protocols are?",
    "Does <Vendor/App> communicate privacy notices, permissions, privacy policies, Terms of Service, contracts etc., in clear, specific and unambiguous language?",
    "Does <Vendor/App> documentation explain to Users how their personal information is being used, processed, disclosed and retained by the provider and any third parties?",
    "Does <Vendor/App> make links to permissions, privacy policies and Terms of Service, etc., easy to find after a User account has been created?",
    "Does <Vendor/App> identify the third parties to which they disclose personal information for processing?",
    "Does <Vendor/App> identify the specific data elements they provide to third-parties, and a summary of User-data protections in place?",
    "Does <Vendor/App> state whether or not it uses User data for statistical analysis and profiling, for making subjective assessments, for predicting behaviour, or as part of a decision-making process?",
    "Does <Vendor/App> provide Users with a mechanism to challenge assessments made based on statistical analysis, profiling, subjective judgements or internal decisions that affect the User?",
    "Does <Vendor/App> state whether or not the business app/service allows users to make personal information publicly available online?",
    "Does <Vendor/App> directly inform users before changes are made to policies and terms of use?",
    "Does <Vendor/App> provide Users the opportunity to change or delete their data before <Vendor/App> uses it in a manner inconsistent with the initial terms the User was provided?",
    "Does <Vendor/App> disclose the presence and use of third party cookies and provide options for managing them?",
    "Does <Vendor/App> confirm that they comply with all pertinent technological regulations?",
    "Does <Vendor/App> confirm that they comply with all Ontario and Canadian laws?",
    "Does <Vendor/App> provide contact information for an operator who will respond to inquiries and challenges from users about privacy policies, data handling practices, accuracy, completeness and use of personal information?",
    "Does <Vendor/App> provide a mechanism for users to access, correct, erase, and download content they created?",
    "Does <Vendor/App> provide users the right to erasure of their data, including metadata inferences, assessments and profiles?",
    "Does <Vendor/App> charge users to erase their data, including metadata inferences, assessments and profiles?",
    "Does <Vendor/App> ensure that when a User deletes their work in the User's account created by an adminstrator, the User's work is also deleted from the administrators account?",
    "Does <Vendor/App> ensure that users have the ability to delete their own accounts?",
    "Does <Vendor/App> ensure Users retain copyright to their own work if they post it to the <Vendor/App> App or <Vendor/App>'s site?"
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
