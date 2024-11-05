from IPython.display import display, Markdown
import openai
import pandas as pd
from openai import OpenAI
import os

client = OpenAI()

description = """The LabTracker software helps (i) doctors manage the requisition of tests and examinations for patients and (ii) patients book appointments for tests and examinations at a lab. For the remainder of this description, tests and examinations are used interchangeably.
For a requisition, a doctor must provide their numeric practitioner number and signature for verification as well as their full name, their address, and their phone number. The signature is a digital signature, i.e., an image of the actual signature of the doctor. Furthermore, the doctor indicates the date from which the requisition is valid. The requisition must also show the patient?? information including their alpha-numeric health number, first name and last name, date of birth, address, and phone number. A doctor cannot prescribe a test for themselves but can prescribe tests to someone else who is a doctor.
Several tests can be combined on one requisition but only if they belong to the same group of tests. For example, only blood tests can be combined on one requisition or only ultrasound examinations can be combined. It is not possible to have a blood test and an ultrasound examination on the same requisition. For each test, its duration is defined by the lab network, so that it is possible to schedule appointments accordingly. The duration of a test is the same at each lab. For some kinds of tests, it does not matter how many tests are performed. They take as long as a single test. For example, several blood tests can be performed on a blood sample, i.e., it takes as long to draw the blood sample for a single blood test as it does for several blood tests.
A doctor may also indicate that the tests on a requisition are to be repeated for a specified number of times and interval. The interval is either weekly, monthly, every half year, or yearly. All tests on a requisition are following the same repetition pattern.
The doctor and the patient can view the results of each test (either negative or positive) as well as the accompanying report.
A patient is required to make an appointment for some tests while others are walk-in only. For example, x-ray examinations require an appointment, but blood tests are walk-in only (i.e., it is not possible to make an appointment for a blood test). On the other hand, some tests only require a sample to be dropped off (e.g., a urine or stool sample).
To make an appointment for a requisition, a patient selects the desired lab based on the lab?? address and business hours. For requisitions with repeated tests, a patient is only allowed to make one appointment at a time. The confirmation for an appointment also shows a confirmation number, the date as well as start/end times, and the name of the lab as well as its registration number. It is possible to change or cancel an appointment at any time but doing so within 24 hours of the appointment incurs a change/cancellation fee. Each lab determines its own fee and business hours. All labs are open every day of the year and offer all tests. The business hours of a lab do not change from one week to the next. Each day a lab is open from the day?? start time to its end time, i.e., there are no breaks.
"""

task_description = """
You are a domain modeling expert and are assigned with the task of domain modeling creation.
You objective is to create a textual based domain modeling given the program description.
There are steps involved in the process. Follow the instruction for your current step.
"""

def noun_analysis(problem_description):
    noun_analysis_prompt = """
    Identify all the nouns in the description which can potentially be the class name, attribute name, role name.
    Include as much as nouns as possible and do not care about their functions for now.
    """
    format_description = """
    only output nouns and separated by , do not include any other words or symbels in your generated text.
    """
    response = client.chat.completions.create(
      model="gpt-4",
      messages=[
          {"role": "system", "content": task_description},
          {"role": "system", "content": problem_description},
          {"role": "system", "content": noun_analysis_prompt},
          {"role": "system", "content": format_description},
      ],
      temperature=0,
    )
    generated_text = response.choices[0].message.content

    noun_list = generated_text.split(",")
    noun_list = [i for i in noun_list if (i != "" and i != "\n" and i != None)]

    return noun_list

def identify_classes(problem_description, nouns_list):
    identify_classes_prompt = """
    Identify classes from the nouns list extracted from the problem description above.
    A class is the description for a set of similar objects that have the same structure and behavior, i.e., its instances
    All objects with the same features and behavior are instances of one class.
    In general, something should be a class if it could have instances.
    In general, something should be an instance if it is clearly a single member of the set defined by a class.
    Keep in mind that some of the nouns may be attributes or roles of the identified classes.
    Choose proper names for classes according the the following rules:
    1. Noun
    2. Singular
    3. Not too general, not too specific – at the right level of abstraction
    4. Avoid software engineering terms (data, record, table, information)
    5. Conventions: first letter capitalized; camel case without spaces if needed

    Example class names:
    Hospital, Doctor, PartTimeEmployee

    Constraints:
    Create classes at the right level of abstraction.
    Not all nouns in the nouns list are classes, some of them may be attributes, role names, or even not needed for diagram.
    Do NOT include all the nouns list as classes. Evaluate if it is needed to be a class.
    ONLY generate classes that are necessary to develop the system.


    Example:
    Problem Description: This system helps the Java Valley police officers keep track of the cases they are assigned to do. Officers may be assigned to investigate particular crimes, which involves interviewing victims at their homes and entering notes in the PI system.
    Identified Class List: PISystem, PoliceStation, Case, PoliceOfficer, Victim, Crime, Note
    """
    format_description = """
    only output class names and separated by , do not include any other words or symbols in your generated text.
    """
    response = client.chat.completions.create(
      model="gpt-4",
      messages=[
          {"role": "system", "content": f"Task Description: {task_description}"},
          {"role": "system", "content": f"Problem Description: {problem_description}"},
          {"role": "system", "content": f"Nouns list: {nouns_list}"},
          {"role": "system", "content": identify_classes_prompt},
          {"role": "system", "content": format_description},
          {"role": "system", "content": f"Identified Class List: \n"},
      ],
      temperature=0,
    )
    generated_text = response.choices[0].message.content

    class_list = generated_text.split(",")
    class_list = [i.strip() for i in class_list if (i != "" and i != "\n" and i != None)]

    return class_list

def identify_attributes(problem_description, class_list, nouns_list):
    identify_attribute_prompt = """
    Given the current identify class list and noun list for potential class, attributes, role names.
    Identify attributes for each class.
    An attribute is a simple piece of data with a name and primitative datatype: string, int, date, time, boolean, etc
    More complex data is NOT modeled as an attribute.
    Attribute exists only when the object of the class exists.
    Conventions: first letter lower case; camel case without spaces if needed

    Notes:
    For each class, evaluate if it can be represented by an attrbute inside another class. If so, remove the class and make it an attribute.
    Do not include the class if it is not necessary in the software system.
    """
    format_description = """
    Follow the format for each class with its attribute: ClassName(type attributeName1, type attributeName2)
    For example:
    Person(string name, string address)
    only output class with attribute in () and separated by each line. do not include any other words or symbels in your generated text.
    """
    constraint = """
    You can overwrite the current class list if some classes are not necessary or should be attributes instead.
    Only generate attributes for the current classes.
    """
    response = client.chat.completions.create(
      model="gpt-4",
      messages=[
          {"role": "system", "content": f"Task Description: {task_description}"},
          {"role": "system", "content": f"Problem Description: {problem_description}"},
          {"role": "system", "content": f"Class list: {class_list}"},
          {"role": "system", "content": f"Noun list: {nouns_list}"},
          {"role": "system", "content": identify_attribute_prompt},
          {"role": "system", "content": constraint},
          {"role": "system", "content": format_description},
      ],
      temperature=0,
    )
    generated_text = response.choices[0].message.content

    class_attribute_list = generated_text.split("\n")
    class_attribute_list = [i for i in class_attribute_list if (i != "" and i != "\n" and i != None)]

    return class_attribute_list


def identify_enumeration_classes(problem_description, class_list, nouns_list):
    identify_classes_prompt = """
    Identify enumeration classes from the current class.
    An enumeration class specifies a predefined list of choices, known as literals.
    Use the keyword "enum" to represent the class is an enumeration class
    For each literal, it consists of mainly one word, without any type.
    Do not show association with an enumeration, indicate as type of attribute.
    Often, the enumeration is defined as a single class, but is referenced for each of the class that needs the enumeration.
    In this case, it is used as an attribute, with the lower case of class name as attribute name and class name as attribute type.

    for example:
    enum PatronType(Student, Adult, Senior)
    LibyaryPatron(PatronType patronType)
    """

    format_description = """
    Follow the format for each class with its attribute: ClassName(type attributeName1, type attributeName2)
    Follow the format for each enumeration class with its literal: enum ClassName(Literal1, Literal2)
    For example:
    Person(string name, string address)
    enum Cake(WeddingCake, BirthdayCake)

    only output class with attribute in () and separated by each line. do not include any other words or symbels in your generated text.
    """

    constraint = """
    Only add the keyword enum if the original class should be an enumeration class
    Output all classes, including enumeration class and normal class
    """

    response = client.chat.completions.create(
      model="gpt-4",
      messages=[
          {"role": "system", "content": task_description},
          {"role": "system", "content": problem_description},
          {"role": "system", "content": f"Class list: {class_list}"},
          {"role": "system", "content": f"Nouns list: {nouns_list}"},
          {"role": "system", "content": identify_classes_prompt},
          {"role": "system", "content": constraint},
          {"role": "system", "content": format_description},
      ],
      temperature=0,
    )
    generated_text = response.choices[0].message.content

    class_list = generated_text.split("\n")
    class_list = [i.strip() for i in class_list if (i != "" and i != "\n" and i != None)]

    return class_list

def identify_abstract_classes(problem_description, class_list, nouns_list):
    identify_classes_prompt = """
    Identify abstract classes from the current class.
    Abstract classes cannot be instantiated, i.e. the object of such class cannot be created directly using the new keyword

    We can treat an abstract class as a superclass and extend it:
    Structure and behavior specified for a superclass also applies to the subclass
    Subclass inherits from superclass

    for example:
    abstract Cake(int price)
    BirthdayCake(int numberOfCandles)
    WeddingCake(int numberOfTiers)

    for example:
    abstract Account(int balance, date openedDate, int creditorOverdraftLimit)
    MortgageAccount(int collateralValue)
    SavingsAccount()
    checkingAccount(int highestCheckNumber)

    use the keyword "abstract" to represent the class is abstract
    """

    format_description = """
    Follow the format for each class with its attribute: ClassName(type attributeName1, type attributeName2)
    For example:
    Person(string name, string address)
    abstract Account(int amount)
    only output class with attribute in () and separated by each line. do not include any other words or symbels in your generated text.
    """

    constraint = """
    Only add the keyword abstract if the original class should be an abstract class
    You can adjust the attributes within the subclass if the super class already contain the attribute
    Output all classes, including abstract classes, normal classes, and enumeration class
    """

    response = client.chat.completions.create(
      model="gpt-4",
      messages=[
          {"role": "system", "content": task_description},
          {"role": "system", "content": problem_description},
          {"role": "system", "content": f"Class list: {class_list}"},
          {"role": "system", "content": f"Nouns list: {nouns_list}"},
          {"role": "system", "content": identify_classes_prompt},
          {"role": "system", "content": constraint},
          {"role": "system", "content": format_description},
      ],
      temperature=0,
    )
    generated_text = response.choices[0].message.content

    class_list = generated_text.split("\n")
    class_list = [i.strip() for i in class_list if (i != "" and i != "\n" and i != None)]

    return class_list

def get_partial_model(problem_description):

  # step 1. Noun analysis
  noun_list = noun_analysis(problem_description)

  # step 2. Identify classes and choose propoer names for the class
  class_list = identify_classes(problem_description, noun_list)

  # step 3. Identify Attributes for each class
  class_attribute_list = identify_attributes(problem_description, class_list, noun_list)

  # step 4. Identify Enumeration class
  class_attribute_enum_list = identify_enumeration_classes(problem_description, class_attribute_list, noun_list)

  # step 5. Identify abstract class
  partial_model = identify_abstract_classes(problem_description, class_attribute_enum_list, noun_list)

  return partial_model

def identify_player_role_pattern_experiment(description, class_attribute):
    identify_classes_prompt = """
    identify the Player-Role pattern within the current classes
    for the classes Student, FullTimeStudent, PartTimeStudent, with the normal super class and subclass relationship,
    an instance of the Student cannot switch from FullTimeStudent to PartTimeStudent, as the instance cannot change type.
    So we need the player role pattern as following:

    Student(string name, string id)
    abstract AttendanceRole()
    FullTimeStudent(int fullTimeCredit) inherit AttendenceRole()
    PartTimeStudent(int partTimeCredit) inherit AttendenceRole()

    Here are more examples:

    Example 1. Within the school system, the student has two roles, graduate student and undergraduate student.
    The student can be a undergrad student at some point, and then switch to the role of graduate student.
    The student class saved information shared by both roles and is associated to the LevelRole.
    Both GraduateStudent and UndergradStudent inherit from the LevelRole class.

    Student(string name)
    abstract LevelRole()
    GraduateStudent(float graduateGpa) inherit LevelRole()
    UndergradStudentfloat undergradGpa inherit LevelRole()

    Example 2. Within the company system, each person has two roles, employee and manager.
    The Person can be an employee at some point, and then switch to the manager role later.

    Person(string name, string email, string address)
    abstract PersonRole()
    Employee(string employeeID) inherit PersonRole()
    Manager(string title) inherit PersonRole()

    Example 3. Within the system, each user has two roles, administrator and player.
    The user can be an administrator at some point, and then switch to the player role, or each user can have two roles at the same time.

    User(string userEmail, string userId)
    abstract UserRole()
    Administrator(string adminName, string adminPassword) inherit UserRole()
    Player(string playAccountName) inherit UserRole()

    Example 4. Within the conference system, each user has three roles: author, program chair, and reviewer.
    The user can have 1-3 roles at the same time. For example, the user can publish a paper as the author, work as a program chair, and review other papers at the same time.

    User(string username, string password)
    abstract UserRole()
    AuthorRole(string authorId) inherit UserRole()
    ProgramChairRole(string programCategory) inherit UserRole()
    ReviewerRole(string averageRating) inherit UserRole()

    Example 5. Within the company system, each person has at most 2 roles: a client, and an employee.
    The person can switch from a client to an employee, or keep two roles at the same time.
    For the role of employee, there are two types, lawer and low clerk. Both roles inherit from the employee role.

    Person(string name, string email)
    abstract UserRole()
    Client(string slientId) inherit UserRole()
    abstract Employee(string employeeId) inherit UserRole()
    Lawyer(string layerCategory) inherit Employee()
    LawClerk(string level) inherit Employee()
    """

    constraint = """
    Only output the classes that are within the Player-Role pattern.
    Do NOT include other classes.
    You may add new classes only if they are part of the Player-Role pattern.
    If there isn't any Player-Role pattern, simply say "No Player-Role pattern identified"
    Only generate Player-Role pattern within the description. Do not repeat the example.
    Ony use the Player-Role pattern when necessary according to the description.
    """

    format_description = """
    Follow the format for each class with its attribute: ClassName(type attributeName1, type attributeName2)
    Use the keyword "abstract" to represent the abstract class
    Use the keyword "inherit" to represent the subclass inherit attributes and relations from the super class
    """

    response = client.chat.completions.create(
      model="gpt-4",
      messages=[
          {"role": "system", "content": task_description},
          {"role": "system", "content": description},
          {"role": "system", "content": f"Class and attribute list: {class_attribute}"},
          {"role": "system", "content": identify_classes_prompt},
          {"role": "system", "content": constraint},
          {"role": "system", "content": format_description},
      ],
      temperature=0.7,
    )
    generated_text = response.choices[0].message.content

    class_list = generated_text.split("\n")
    class_list = [i.strip() for i in class_list if (i != "" and i != "\n" and i != None)]

    return class_list

def summarize_player_role_pattern(description, result_list, class_attribute):
  summarize_prompt = """Identify the Player-Role pattern from the descriotion provided with reference to five result list.
  Output the mostly like Player-Role pattern according to 5 result you have.
  You do not need to included everything from the 5 result you have, only include the classes you think it is correct.
  Combine the 5 result you have and make the final solution that make sense to you.
  Do not output other classes that are not included in the Player-Role pattern.
  If there isn't any Player-Role pattern, simply say "No Player-Role pattern identified"
  """

  format_description = """
  Follow the format for each class with its attribute: ClassName(type attributeName1, type attributeName2)
  Use the keyword "abstract" to represent the abstract class
  Use the keyword "inherit" to represent the subclass inherit attributes and relations from the super class.
  for example:
  Person(string name, string email, string address)
  abstract PersonRole()
  Employee(string employeeID) inherit PersonRole()
  Manager(string title) inherit PersonRole()
  """

  response = client.chat.completions.create(
  model="gpt-4",
  messages=[
      {"role": "system", "content": f"Task description: {task_description}"},
      {"role": "system", "content": summarize_prompt},
      {"role": "system", "content": f"Description: {description}"},
      {"role": "system", "content": f"5 solution list {result_list}"},
      {"role": "system", "content": f"Class and attribute list: {class_attribute}"},
      {"role": "system", "content": f"Format description: {format_description}"},
  ],
  temperature=0,
  )
  generated_text = response.choices[0].message.content
  return generated_text

def class_integrater(description, generated_class_list, player_role_pattern):
  checker_prompt = """Using the current generated classes and identified player role pattern,
  combine the two versions and generate the final version of classes.

  Do the following things:
  1. analysis the generated classes to see if they are needed.
  Some generated classes may not be the right level of abstraction.
  Drop the classes if there are not necessary to describe the system.
  2. evaluate the player-role pattern to see if they are necessary.
  Not all system need the player-role pattern.
  Since player-role pattern can be complex in implementation, only use it if it is necessary.
  if the abstract classes and their subclasses are necessary, do not use player-role pattern.
  3. Combine the two version and make a solution that is consistent with both versions.
  Do not have duplicate classes in the final solution
  """

  format_description = """
  Do not generate other phrases besides the classes.
  Do not generate number for the classes.
  Follow the format for each class with its attribute: ClassName(type attributeName1, type attributeName2)
  Use the keyword "abstract" to represent the abstract class
  Use the keyword "inherit" to represent the subclass inherit attributes and relations from the super class.
  for example:
  Person(string name, string email, string address)
  abstract PersonRole()
  Employee(string employeeID) inherit PersonRole()
  Manager(string title) inherit PersonRole()
  """

  response = client.chat.completions.create(
  model="gpt-4",
  messages=[
      {"role": "system", "content": f"Task description: {task_description}"},
      {"role": "system", "content": checker_prompt},
      {"role": "system", "content": f"Description: {description}"},
      {"role": "system", "content": f"Generated classes list {generated_class_list}"},
      {"role": "system", "content": f"Player-role pattern: {player_role_pattern}"},
      {"role": "system", "content": f"Format description: {format_description}"},
      {"role": "system", "content": f"Integrated classes with attributes: \n"},
  ],
  temperature=0,
  )
  generated_text = response.choices[0].message.content

  class_list = generated_text.split("\n")
  class_list = [i.strip() for i in class_list if (i != "" and i != "\n" and i != None)]

  return class_list


def checker(problem_description, class_list):
    checker_prompt = """
    Given the class list for the problem description, write comment for each class with its attribute.
    Evaluate if it is at the correct level of abstraction to be included in the software system.
    Many classes may not be needed and may not be necessary, example cases:
    - if class A is too detailed to be included in the system, consider removing it.
    - if class A does not contain any attributes or only contains 1 attribute, consider moving the attribute of class A to another class and removing class A
    - For the enumeration class, evaluate if it should be captured by an attribute and if its literals are necessary
    - For the subclasses, evaluate if they are necessary to be present in the system.

    You can write general comments and comments to each class, evaluate if the class is necessary. If not, provide a solution to change it.
    """
    response = client.chat.completions.create(
      model="gpt-4",
      messages=[
          {"role": "system", "content": f"Task Description: {task_description}"},
          {"role": "system", "content": f"Problem Description: {problem_description}"},
          {"role": "system", "content": f"Class list: {class_list}"},
          {"role": "system", "content": checker_prompt},
          {"role": "system", "content": f"Generated comments: \n"},
      ],
      temperature=0,
    )
    generated_text = response.choices[0].message.content

    class_attribute_list = generated_text.split("\n")
    class_attribute_list = [i for i in class_attribute_list if (i != "" and i != "\n" and i != None)]

    return class_attribute_list

def integrate_feedback_from_checker(problem_description, class_attribute_list, checker_comment):
  integrate_prompt = """
  integrate the feedback given by the checker to finish the class diagram according to the problem description.
  """

  format_description = """
  Do not generate other phrases besides the classes.
  Do not generate number for the classes.
  Follow the format for each class with its attribute: ClassName(type attributeName1, type attributeName2)
  For example:
  Person(string name, string address)
  only output class with attribute in () and separated by each line. do not include any other words or symbels in your generated text.
  """

  response = client.chat.completions.create(
  model="gpt-4",
  messages=[
      {"role": "system", "content": f"Task Description: {task_description}"},
      {"role": "system", "content": f"Problem Description: {problem_description}"},
      {"role": "system", "content": f"Class list: {class_attribute_list}"},
      {"role": "system", "content": f"Feedback from checker: {checker_comment}"},
      {"role": "system", "content": integrate_prompt},
      {"role": "system", "content": format_description},
      {"role": "system", "content": "Revised class diagram: "},
  ],
  temperature=0,
  )
  generated_text = response.choices[0].message.content

  class_attribute_list = generated_text.split("\n")
  class_attribute_list = [i for i in class_attribute_list if (i != "" and i != "\n" and i != None)]

  return class_attribute_list

def identify_relationship(description, class_list, player_role):
  relationship_prompt = """
  Identify relationships between classes. There are three types of relationships:

  1. Composition with the keyword "contain"
  example format: mul1 Class1 contain mul2 Class2
  Class1 and Class2 are classes above. mul1 and mul2 are one of the following options[0..*, 1, 0..1, 1..*]
  there might be multiple compositions
  In a typical domain model, there is usually a "system class" that contain most of the classes within the system
  For example:
  1 SchoolSystem contain 0..* UserRole
  1 SchoolSystem contain 0..* User
  1 SchoolSystem contain 0..* Course
  1 SchoolSystem contain 0..* Registration
  1 SchoolSystem contain 0..* StudentProfile

  2. Inheritance with the keyword "inherit"
  example format: Class1 inherit Class2
  Class1 and Class2 are classes above. there might be multiple inheritance
  Consider the inheritance relationship within the Player-Role pattern
  For example:
  Student inherit PersonRole
  Professor inherit PersonRole

  3. Association with the keyword "associate"
  example format: mul1 Class1 associate mul2 Class2
  Class1 and Class2 are classes above. mul1 and mul2 are one of the following options[0..*, 1, 0..1, 1..*]
  there might be multiple associations
  For example:
  0..* Student associate 0..5 Registration
  1 Student associate 0..1 StudentProfile

  Note:
  1. Use the classes in the given generated classes list, generate the classes and their relationships.
  2. Only add the system class if the existing class diagram misses the system class.
  3. Do NOT change existing classes or add other classes besides the system class.
  4. In most of the cases, there is only 1 relationship within the same two classes.
  """

  format_description = """
  Generate the complete class diagram according to the class list using the following format:

  Classes:
  <put the origianl input class list here, do not modify existing classes>

  Relatipnships:
  Composition:
  <put composition relationship here using the format: mul1 Class1 contain mul2 Class2>
  Inheritance:
  <put inheritance relationship here using the format: Class1 inherit Class2>
  Association:
  <put association relationship here using the format: mul1 Class1 associate mul2 Class2>

  Make sure the generated text can be processed by text.split("\n") and then [text.strip()] into a list of processed classes and relationships
  """

  response = client.chat.completions.create(
  model="gpt-4",
  messages=[
      {"role": "system", "content": relationship_prompt},
      {"role": "system", "content": f"Description: {description}"},
      {"role": "system", "content": f"Generated classes list: {class_list}"},
      {"role": "system", "content": f"Player-role pattern: {player_role}"},
      {"role": "system", "content": f"Format description: {format_description}"},
  ],
  temperature=0,
  )
  generated_text = response.choices[0].message.content

  complete_class_diagram = generated_text.split("\n")
  complete_class_diagram = [i.strip() for i in complete_class_diagram if (i != "" and i != "\n" and i != None)]

  return complete_class_diagram

def domain_model_generation(problem_description):
  # step 1 identify classes and attributes
  partial_model = get_partial_model(problem_description)

  # step 2 identify the player-role pattern
  iteration_list = []
  for i in range(1):
    player_role_pattern = identify_player_role_pattern_experiment(problem_description, partial_model)
    iteration_list.append(player_role_pattern)

  player_role_pattern = summarize_player_role_pattern(problem_description, iteration_list, partial_model)

  completed_class_diagram = class_integrater(problem_description, partial_model, player_role_pattern)

  # step 3 self-reflection
  comment = checker(problem_description, completed_class_diagram)
  revised_result = integrate_feedback_from_checker(problem_description, completed_class_diagram, comment)

  # step 4 identify relationships
  final_result = identify_relationship(problem_description, revised_result, player_role_pattern)

  return final_result