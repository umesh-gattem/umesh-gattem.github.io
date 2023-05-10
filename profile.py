# Introduction

def introduction(profile):
    umesh_introduction = """Greetings, my name is Umesh Kumar, and I am an enthusiastic developer and designer with extensive
                experience in the field of Artificial Intelligence and Deep Learning technologies. I have worked with
                various backends such as TensorFlow, Keras, and PyTorch. My proficiency in Python programming spans
                over seven years. As a highly motivated AI engineer, I have been involved in the design of tools for
                visually constructing data transformation recipes, developing Deep Learning models, and end-to-end
                pipelines for Data Scientists.<br><br>
                I am an excellent team player who enjoys contributing to product development by taking proactive
                initiatives and assuming responsibility for projects. I am excited to apply my skills and expertise
                to real-world projects that have the potential to positively impact thousands of people and make
                their lives easier."""
    umesh_introduction = profile.replace("##### UMESH INTRODUCTION", umesh_introduction)
    return umesh_introduction


def experience(profile):
    associate_instructor = """Recently, I served as a Teaching Assistant or Associate Instructor for the Department of 
    Statistics at IUB, specifically for the Applied Statistical Computing Course.\n\n 
    The course curriculum covers several topics, including Statistical Computing using R and C/C++, among others. \n\n
    In the past, I served as a Teaching Assistant or Associate Instructor for the Introduction to Statistics Course 
    offered by the Department of Statistics at IUB. \n\n
    The course covered several topics, including Probability Models, Statistical Methods, Maximum Likelihood, 
    Method of Least Squares, and distribution functions, among others. \n\n
    As part of my responsibilities, I conduct weekly office hours to assist and clarify students' doubts and questions 
    related to the course. \n\n
    Additionally, I supervise graduate and PhD students by creating assignment questions, grading their work, 
    providing feedback, and offering solutions to their queries.
    """

    razorthink = """
    I was part of the Razorthink AI Platform Product team, responsible for developing a tool that 
    enables data scientists and analysts to visually construct data transformation recipes, deep learning models, 
    and end-to-end pipelines. \n\n
    In my role, I designed various modeling libraries using cutting-edge technologies such as Transfer Learning, 
    Training and Inferring models, Tensorboard, TFHUB models, and Distributed Training, 
    exploratory data analysis (EDA), and end-to-end pipelines\n\n
    Moreover, I developed the RZTDL Library, a patented deep learning framework that leverages different 
    backends of Tensorflow and Pytorch to support all Deep Learning operations like CNN, RNN, LSTM, Attention, etc. \n\n
    Designed a multi-stage CNN model for analyzing multidimensional time-series data, resulting in a 
    GINI score of 72 forpredicting customer propensity to buy insurance for our esteemed banking clients.\n\n
    Utilized LSTM network to develop a churn prediction model by conducting a comprehensive analysis of 
    demographics and skewed transactional data, resulting in a GINI score of 68 for one of our 
    largest telecommunications clients.\n\n
    Led a team of skilled professionals in developing a Python SDK, which facilitated the generation of blueprints, 
    automated the process of generating code, and created APIs for seamless integration with web applications.
    """
    IUB_experience = ""
    razorthink_experience = ""
    for responsibility in associate_instructor.split("\n\n"):
        IUB_experience += '<li style="text-align:justify">' + responsibility + '</li>'
    for responsibility in razorthink.split("\n\n"):
        razorthink_experience += '<li style="text-align:justify">' + responsibility + '</li>'
    profile = profile.replace("##### UMESH IUB EXPERIENCE", IUB_experience)
    umesh_experience = profile.replace("##### UMESH RAZORTHINK EXPERIENCE", razorthink_experience)
    return umesh_experience


def projects(profile):
    cv_project = """
    The aim of this project is to construct a personalized image classification system for Google Photos by 
    applying computer vision techniques based on deep learning. \n\n
    Rather than relying on traditional facial recognition algorithms, this project intends to use a custom dataset 
    and implement transfer learning to extract high-level features from pre-trained models like Inception and ResNet.\n\n 
    To address the challenge of limited data, the project will employ various data preprocessing techniques such as 
    data augmentation, cropping, and rotation, and may also leverage advanced deep learning models like 
    Generative Adversarial Networks (GAN) or Autoencoders to generate additional data.\n\n 
    Ultimately, the goal is to develop a customized model that can accurately classify images 
    within a user's Google Photos collection. \n\n
    The project also seeks to create a comprehensive pipeline for the Google Photos System that can perform person 
    detection and recognition, utilizing the YOLO (You Only Look Once) object detection model to identify 
    faces of individuals by repurposing it from the COCO dataset. \n\n
    The system will be capable of analyzing both individual and group photos, identifying all faces in the image, 
    creating boundary boxes around each person, and correctly labeling each individual with their corresponding name.
    """
    mtgnn = """
    Our research focused on exploring MT-GNN, a cutting-edge model for graph learning and modeling spatio-temporal 
    information. To evaluate its effectiveness, we compared its results with several baseline models we constructed. \n\n 
    MT-GNN is a highly complex deep learning model, comprising multiple components that interact with each other.
    In order to gain a better understanding of the model, we invested significant time in studying it and reviewing 
    the official code that was provided along with the paper.  \n\n
    We conducted empirical analysis on multiple datasets, including Traffic Dataset, Solar Energy Dataset, Pems-D7, 
    Paris Mobility, and Energy Consumption, and presented our findings.
    """
    nlp_project = """
    In this project, my objective was to detect the boundary boxes of text in a given document or image file. 
    To accomplish this, I employed several libraries and compared their respective results.  \n\n
    Notably, I utilized technologies such as PaddleOCR, Tesseract OCR, Google API for image recognition, and 
    Layout Language Model.  \n\n
    To evaluate the effectiveness of these libraries, I used a dataset from Wantok images, which is a rare 
    language with a complicated format that makes it challenging to predict the text based on the layout and 
    structure of these images. Despite these difficulties, I successfully extracted and recognized the text 
    boundary boxes with satisfactory accuracy.
    """
    ipl_d_viz = """
    As a part of my Data Visualization course, I worked on visualizing the IPL Dataset.  \n\n
    This dataset contains the statistics of all IPL teams and players across all IPL leagues that 
    have taken place until now. \n\n
    My project involves creating various visualizations, including but not limited to: 'Team performance at 
    home ground', 'Team performance away from home ground', 'Best performance at a non-home ground', 
    'Strike rates and averages of batsmen', 'Strike rates and averages of bowlers', 'Batsman performance 
    against specific bowlers', 'Bowler performance against particular batsmen', and potentially many more.
    """
    rztdl = """
    RZTDL is a cutting-edge deep learning framework that is patented and developed using various 
    backends such as Tensorflow and Pytorch.  \n\n
    It has a wide range of features such as distributed training, transfer learning, Tensorboard, and TF Hub Models.  \n\n
    As part of my responsibilities, I was in charge of supporting all Tensorflow operations like Layers, 
    Operators, Metrics, etc.  \n\n
    I was also responsible for implementing backend APIs for training and inferring deep learning models.  \n\n
    Additionally, I played a key role in the development, testing, and deployment of production solutions.
    """
    python_parser = """
    During my time at Razorthink AI, I was involved in various Python SDK projects related to the Razorthink 
    AI product.  \n\n
    One of my responsibilities included creating Python libraries that contained parsing logics for 
    generating blueprints for different Tensorflow layers.  \n\n
    These blueprints enabled users to easily drag and drop the required blocks, specify parameters and then 
    convert the JSON code to the required Tensorflow code, or vice versa, for training their models. \n\n
    Additionally, I had the opportunity to work on Micro web services using Flask Python and swagger-api-client modules.
    """

    cv_project_details = ""
    mtgnn_details = ""
    nlp_project_details = ""
    ipl_d_viz_details = ""
    rztdl_details = ""
    parser_details = ""
    for responsibility in cv_project.split("\n\n"):
        cv_project_details += '<li style="text-align:justify">' + responsibility + '</li>'
    for responsibility in mtgnn.split("\n\n"):
        mtgnn_details += '<li style="text-align:justify">' + responsibility + '</li>'
    for responsibility in nlp_project.split("\n\n"):
        nlp_project_details += '<li style="text-align:justify">' + responsibility + '</li>'
    for responsibility in ipl_d_viz.split("\n\n"):
        ipl_d_viz_details += '<li style="text-align:justify">' + responsibility + '</li>'
    for responsibility in rztdl.split("\n\n"):
        rztdl_details += '<li style="text-align:justify">' + responsibility + '</li>'
    for responsibility in python_parser.split("\n\n"):
        parser_details += '<li style="text-align:justify">' + responsibility + '</li>'
    profile = profile.replace("##### UMESH CV PROJECT", cv_project_details)
    profile = profile.replace("##### UMESH MTGNN", mtgnn_details)
    profile = profile.replace("##### UMESH NLP PROJECT", nlp_project_details)
    profile = profile.replace("##### UMESH IPL D VIZ", ipl_d_viz_details)
    profile = profile.replace("##### UMESH RZTDL", rztdl_details)
    projects = profile.replace("##### UMESH PARSER", parser_details)
    return projects


def skills(profile):
    skills_dict = {"Frameworks": "TensorFlow, PyTorch, Keras, Sci - kit, Pandas, Numpy, Flask, OpenCV, "
                                 "Pyspark, FastAPI, Uvicorn",
                   "Machine Learning": "PCA, T - SNE, TFHub, Neural Networks, Clustering, "
                                       "Transfer Learning, Inferring models",
                   "Databases": "MySQL, Postgres, MongoDB, SQLite",
                   "Big Data": "Spark, Hadoop, Kafka, S3, Horovod",
                   "MLOps": "Docker, Kubernetes, GCP, AWS, Weights and Bias, Distributed Training, Tensorboard",
                   "Project Management and Tools": "Git, JIRA, Confluence, Slack, Pycharm, IntelliJ, Jupyter Notebook"}
    skills = ""
    for key, skill in skills_dict.items():
        skills += f'<li style="text-align:justify"><strong>{key}</strong> - '
        skills += skill + "</li>"
    skills = profile.replace("##### UMESH SKILLS", skills)
    return skills


def interests(profile):
    umesh_interests = """
    I possess a deep passion for coding, mathematics, and logical thinking. Alongside my technical abilities, 
    I have honed my teaching skills through training freshers in my previous company and teaching Python to friends. 
    I actively participate in coding competitions on platforms such as Codechef and Leetcode, seeking to continuously 
    enhance my skills.<br><br>
    In addition to my interest in technology, I am an avid cricket fan, closely following the ICC and BCCI, as well 
    as other sporting events around the world. During my free time, I indulge in watching TV series, and thanks to 
    the pandemic, I have discovered and enjoyed a multitude of new shows."""
    umesh_interests = profile.replace("##### UMESH INTERESTS", umesh_interests)
    return umesh_interests


if __name__ == '__main__':
    with open("index.txt", "r") as file:
        index = file.read()

    updated_profile = introduction(index)
    updated_profile = experience(updated_profile)
    updated_profile = projects(updated_profile)
    updated_profile = skills(updated_profile)
    updated_profile = interests(updated_profile)

    with open("index.html", 'w') as sample:
        sample.write(updated_profile)
