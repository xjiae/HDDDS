# Ground Truth eXplanation Dataset (GTX)

The Ground Truth eXplanation (GTX) dataset is a curated collection that addresses the challenge of evaluating the quality of explainability methods. Existing approaches often lack ground truth explanations and heavily rely on hand-crafted heuristics. In response, the GTX dataset has been created to assess the alignment of feature attributions with human annotations. It contains time-series data (HAI, SWaT, WADI) from the industrial control domain, image data (MVTec) from the defect inspection domain, and text data (SQuAD) from the machine comprehension domain.

#### Dataset Link
<!-- info: Provide a link to the dataset: -->
<!-- width: half -->
Dataset Link: [HAI](https://github.com/icsdataset/hai), [SWaT, WADI](https://itrust.sutd.edu.sg/itrust-labs_datasets/), [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad), [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/).

#### Data Card Author(s)
<!-- info: Select **one role per** Data Card Author:

(Usage Note: Select the most appropriate choice to describe the author's role
in creating the Data Card.) -->
<!-- width: half -->
- **Xiayan Ji, University of Pennsylvania:** (Manager)
- **Anton Xue, University of Pennsylvania:** (Manager)

## Authorship

### Dataset Owners
#### Team(s)
<!-- scope: telescope -->
<!-- info: Provide the names of the groups or team(s) that own the dataset: -->
University of Pennsylvania

#### Author(s)
<!-- scope: microscope -->
<!-- info: Provide the details of all authors associated with the dataset:
(Usage Note: Provide the affiliation and year if different from publishing
institutions or multiple affiliations.) -->
- Xiayan Ji, Ph.D. Student, University of Pennsylvania, 2023
- Anton Xue, Ph.D. Student, University of Pennsylvania, 2023
- Rajeev Alur, Professor, University of Pennsylvania, 2023
- Oleg Sokolsky, Professor, University of Pennsylvania, 2023
- Insup Lee, Professor, University of Pennsylvania, 2023
- Eric Wong, Assistant Professor, University of Pennsylvania, 2023

### Funding Sources
#### Institution(s)
<!-- scope: telescope -->
<!-- info: Provide the names of the funding institution(s): -->
- Name of Institution
- Name of Institution
- Name of Institution

#### Funding or Grant Summary(ies)
<!-- scope: periscope -->
<!-- width: full -->
<!-- info: Provide a short summary of programs or projects that may have funded
the creation, collection, or curation of the dataset.

Use additional notes to capture any other relevant information or
considerations. -->
*For example, Institution 1 and institution 2 jointly funded this dataset as a
part of the XYZ data program, funded by XYZ grant awarded by institution 3 for
the years YYYY-YYYY.*

Summarize here. Link to documents if available.

**Additional Notes:** Add here -->

#### Contact Detail(s)
<!-- scope: periscope -->
<!-- info: Provide pathways to contact dataset owners: -->
- **Point of Contact:** Xiayan Ji
- **Affiliation:** University of Pennsylvania
- **Contact:** xjiae@seas.upenn.edu
## Dataset Overview
#### Data Subject(s)
<!-- scope: telescope -->
<!-- info: Select ***all applicable**** subjects contained the dataset: -->
- Data about places and objects
- Synthetically generated data
- Data about systems or products and their behaviors


#### Dataset Snapshot
<!-- scope: periscope -->
<!-- info: Provide a snapshot of the dataset:<br><br>(Use the additional notes
to include relevant information, considerations, and links to table(s) with
more detailed breakdowns.) -->
Category | Data
--- | ---
Size of Dataset | 12 GB
Number of Instances | 3,798,242
Number of Labels (explanation) | 5,951,278,880
Average Labeles Per Instance | 1566.85
Algorithmic Labels | 4,629,687,370
Human Labels | 1,321,591,510
<!-- Other Characteristics | 123456 -->

**Dataset Summary:** time-series, image and text data with ground truth explanation labels.

<!-- **Additional Notes:** Add here. -->

#### Content Description
<!-- scope: microscope -->
<!-- info: Provide a short description of the content in a data point: -->
Each content contains an input data (x), a target label (y) and an explanation (a).

**Additional Notes:** for SQuAD, the format is slightly different, the input and target are combined together to better be fitted to a language model. In addition, the explanation is in the form of a start and end position.


#### Risk Type(s)
<!-- scope: telescope -->
<!-- info: Select **all applicable** risk types presenting from the
dataset: -->
- No Known Risks

### Dataset Version and Maintenance
#### Maintenance Status
<!-- scope: telescope -->
<!-- info: Select **one:** -->
**Regularly Updated** - New versions of the dataset
have been or will continue to be
made available.

#### Version Details
<!-- scope: periscope -->
<!-- info: Provide details about **this** version of the dataset: -->
**Current Version:** 1.0

**Last Updated:** 06/2023

**Release Date:** 06/2023

#### Maintenance Plan
<!-- scope: microscope -->
<!-- info: Summarize the maintenance plan for the dataset:

Use additional notes to capture any other relevant information or
considerations. -->
In our maintenance plan, our primary focus will be on preserving and leveraging the existing data that we have collected. This involves ensuring the integrity and security of the data through regular backups, implementing robust data storage practices, and conducting periodic audits to identify any potential issues or anomalies. Additionally, we recognize the growing importance of graph datasets in various domains. To capitalize on this, we will actively explore and evaluate potential graph datasets that align with our needs and objectives. This includes seeking out reliable sources, assessing the quality and relevance of the data, and integrating suitable graph datasets into our existing infrastructure. By incorporating graph datasets, we aim to enhance the depth and breadth of our analysis, uncover hidden patterns and relationships, and gain valuable insights that can drive informed decision-making and optimize our operations. Our maintenance plan thus combines the preservation of existing data with the exploration of new graph datasets, ensuring a comprehensive and forward-looking approach to data management and utilization.

**Versioning:** The dataset is versioned based on several criteria. This includes significant updates or changes in the data collection process, methodology, or data sources. Corrections or improvements to enhance data accuracy or reliability also warrant a new version. Substantial additions or expansions, such as new data points or variables, are considered for versioning. User feedback and requests for specific modifications are also taken into account. The versioning process ensures transparency, traceability, and reproducibility, keeping the dataset relevant and adaptable to evolving needs.

**Updates:** The dataset is refreshed or updated based on regular time-based updates, changes in data sources or collection methodologies, user feedback, and advancements in technology or analytical techniques. This ensures the dataset remains relevant, accurate, and valuable for users in making informed decisions.

**Errors:** Error handling for the dataset involves systematic procedures to identify and correct errors, maintaining data integrity through documentation and tracking, and implementing measures to prevent future errors. These criteria ensure data quality, transparency, and reliability for users.

**Feedback:** The dataset incorporates criteria for feedback by actively seeking input from users and stakeholders. Feedback on the dataset's content, quality, and usability is welcomed and considered for future updates and improvements. This iterative feedback process ensures that the dataset meets the needs and expectations of its users, enhancing its relevance and value.


#### Next Planned Update(s)
<!-- scope: periscope -->
<!-- info: Provide details about the next planned update: -->
**Version affected:** 1.0

**Next data update:** 08/2023

**Next version:** 1.1

**Next version update:** 08/2023

#### Expected Change(s)
<!-- scope: microscope -->
<!-- info: Summarize the updates to the dataset and/or data that are expected
on the next update.

Use additional notes to capture any other relevant information or
considerations. -->
**Updates to Data:** Next version of the dataset will possibly include suitable graph dataset.


## Example of Data Points
#### Primary Data Modality
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Image Data
- Text Data
- Time Series

#### Sampling of Data Points
<!-- scope: periscope -->
<!-- info: Provide link(s) to data points or exploratory demos: -->
- [Demo Link](/HDDDS/example.ipynb)

#### Data Fields
<!-- scope: microscope -->
<!-- info: List the fields in data points and their descriptions.

(Usage Note: Describe each field in a data point. Optionally use this to show
the example.) -->

Field Name | Field Value | Description
--- | --- | ---
x | input data | The input data, time-series or image or pagraph.
y| target label (0/1) | The target label of attacked/defect/answerable.
a | explanation | The ground truth feature to explain the target label.

<!-- **Above:** Provide a caption for the above table or visualization if used. -->

<!-- **Additional Notes:** Add here -->

#### Typical Data Point
<!-- width: half -->
<!-- info: Provide an example of a typical data point and describe what makes
it typical.

**Use additional notes to capture any other relevant information or
considerations.** -->
This is a typical data point:

```
{'x': tensor([[0.6273, 0.2893, 0.2775,  ..., 0.4198, 0.3439, 0.5313],
        [0.6273, 0.2985, 0.2775,  ..., 0.4198, 0.3401, 0.5330],
        [0.6273, 0.3055, 0.2775,  ..., 0.4198, 0.3439, 0.5292],
        ...,
        [0.6273, 0.3265, 0.2775,  ..., 0.4198, 0.3467, 0.4995],
        [0.6273, 0.3341, 0.2775,  ..., 0.4198, 0.3467, 0.5019],
        [0.6273, 0.3444, 0.2775,  ..., 0.4198, 0.3467, 0.5022]]),
  'y': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0])
  'a': tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], dtype=torch.float64)}
```

<!-- **Additional Notes:** Add here -->

#### Atypical Data Point
<!-- width: half -->
<!-- info: Provide an example of an outlier data point and describe what makes
it atypical.

**Use additional notes to capture any other relevant information or
considerations.** -->
This is an example for SQuAD dataset:

```
{'q_id': '8houtx',
  'title': 'Why does water heated to room temperature feel colder than the air around it?',
  'selftext': '',
  'document': '',
  'subreddit': 'explainlikeimfive',
  'answers': {'a_id': ['dylcnfk', 'dylcj49'],
  'text': ["Water transfers heat more efficiently than air. When something feels cold it's because heat is being transferred from your skin to whatever you're touching. ... Get out of the water and have a breeze blow on you while you're wet, all of the water starts evaporating, pulling even more heat from you."],
  'score': [5, 2]},
  'title_urls': {'url': []},
  'selftext_urls': {'url': []},
  'answers_urls': {'url': []}}
```

## Motivations & Intentions
### Motivations
#### Purpose(s)
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Research

#### Domain(s) of Application
<!-- scope: periscope -->
<!-- info: Provide a list of key domains of application that the dataset has
been designed for:<br><br>(Usage Note: Use comma-separated keywords.) -->
`Machine Learning`, `Explainability`, `XAI`, `Anomaly Detection` .


#### Motivating Factor(s)
<!-- scope: microscope -->
<!-- info: List the primary motivations for creating or curating this dataset:

(Usage Note: use this to describe the problem space and corresponding
motivations for the dataset.) -->
<!-- For example: -->

- Evaluating the quality of explainability methods is challenging due to the lack of ground truth explanations, and often rely on hand-crafted heuristics. 
- Re-aligning explainable models with human explanations


### Intended Use
#### Dataset Use(s)
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Safe for research use



#### Suitable Use Case(s)
<!-- scope: periscope -->
<!-- info: Summarize known suitable and intended use cases of this dataset.

Use additional notes to capture any specific patterns that readers should
look out for, or other relevant information or considerations. -->
**Suitable Use Case:** One suitable use case for the dataset is in the field of explainable artificial intelligence (AI). The dataset, Ground Truth eXplanation (GTX), provides a valuable resource for evaluating and improving feature attribution methods. Researchers and practitioners in the field can utilize the dataset to benchmark and compare different algorithms, assess their alignment with human annotations, and identify areas for improvement. The diverse nature of the dataset, spanning various data types such as time-series, images, and text, allows for comprehensive evaluation in different real-world scenarios.

<!-- **Additional Notes:** Add here -->

#### Unsuitable Use Case(s)
<!-- scope: microscope -->
<!-- info: Summarize known unsuitable and unintended use cases of this dataset.

Use additional notes to capture any specific patterns that readers should look
out for, or other relevant information or considerations. -->
**Unsuitable Use Case:** **Suitable Use Case:** One suitable use case for the dataset is in the field of explainable artificial intelligence (AI). The dataset, Ground Truth eXplanation (GTX), provides a valuable resource for evaluating and improving feature attribution methods. Researchers and practitioners in the field can utilize the dataset to benchmark and compare different algorithms, assess their alignment with human annotations, and identify areas for improvement. The diverse nature of the dataset, spanning various data types such as time-series, images, and text, allows for comprehensive evaluation in different real-world scenarios.


#### Research and Problem Space(s)
<!-- scope: periscope -->
<!-- info: Provide a description of the specific problem space that this
dataset intends to address. -->
The specific problem space that the Ground Truth eXplanation (GTX) dataset aims to address is the evaluation and improvement of feature attribution methods in explainable artificial intelligence (AI). The dataset seeks to tackle the challenge of assessing the alignment between feature attributions and human annotations, providing a quantitative benchmark for evaluating the quality of these methods.

#### Citation Guidelines
<!-- scope: microscope -->
<!-- info: Provide guidelines and steps for citing this dataset in research
and/or production.

Use additional notes to capture any specific patterns that readers should look
out for, or other relevant information or considerations. -->
**Guidelines & Steps:** Please cite our woek as follows (to be updated later):


**BiBTeX:**
```
@article{snp2023,
  title={Ground Truth eXplanation Datset},
  author={../},
  journal={...},
  year={2023}
}
```



## Access, Rentention, & Wipeout
### Access
#### Access Type
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- External - Open Access


#### Documentation Link(s)
<!-- scope: periscope -->
<!-- info: Provide links that describe documentation to access this
dataset: -->
- [GitHub URL](https://github.com/xjiae/HDDDS)



#### Policy Link(s)
<!-- scope: periscope -->
<!-- info: Provide a link to the access policy: -->
- Direct download URL: [link](https://drive.google.com/uc?id=1oxll8MirhzR3skk0MWpznqeLP9k4Khv9&export=download-anyway&confirm=t&uuid=e5c2b20b-2f3d-4f4f-8e4f-b77d0d32a35d&at=AKKF8vx6FMG8ziRR23ieeLqDfLqN:1684956011585)


Code to download data:
```
wget -O data.zip "https://drive.google.com/uc?id=1oxll8MirhzR3skk0MWpznqeLP9k4Khv9&export=download-anyway&confirm=t&uuid=e5c2b20b-2f3d-4f4f-8e4f-b77d0d32a35d&at=AKKF8vx6FMG8ziRR23ieeLqDfLqN:1684956011585"
unzip data.zip
```



### Retention
#### Duration
<!-- scope: periscope -->
<!-- info: Specify the duration for which this dataset can be retained: -->
Infinite duration.

## Provenance
### Collection
#### Method(s) Used
<!-- scope: telescope -->
<!-- info: Select **all applicable** methods used to collect data: -->

- Taken from other existing datasets


#### Methodology Detail(s)
<!-- scope: periscope -->
<!-- info: Provide a description of each collection method used.

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for collection method
type.) -->
**Collection Type**

**Source:** [HAI](https://github.com/icsdataset/hai), [SWaT, WADI](https://itrust.sutd.edu.sg/itrust-labs_datasets/), [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad), [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/).


**Is this source considered sensitive or high-risk?** [No]

**Dates of Collection:** [05 2023 - 06 2023]

**Primary modality of collection data:**

<!-- *Usage Note: Select one for this collection type.* -->

- Image Data
- Text Data
- Time Series


**Update Frequency for collected data:**

<!-- *Usage Note: Select one for this collection type.* -->


- Static


#### Source Description(s)
<!-- scope: microscope -->
<!-- info: Provide a description of each upstream source of data.

Use additional notes to capture any other relevant information or
considerations. -->
- **Source:** Hardware-In-the-Loop-based Augmented ICS Security Dataset [(HAI)](https://github.com/icsdataset/hai)  The HAI dataset was collected from a realistic industrial control system (ICS) testbed, augmented with a Hardware-In-the-Loop (HIL) simulator for 379.3 hours. The HIL simulator emulates two crucial components of the power generation domain: steam-turbine power generation and pumped-storage hydropower generation, with a total of m = 86 features.
- **Source:** [SWaT, WADI](https://itrust.sutd.edu.sg/itrust-labs_datasets/). The Secure Water Treatment testbed serves as a scaled-down replica of a real-world industrial water treatment plant. It operates at a reduced capacity, producing five gallons per minute of water for over 11 days. The treatment process involves the utilization of membrane-based ultrafiltration and reverse osmosis units for effective water filtration, comprising of \(m = 51\) features in total.   WADI is an extension of the SWaT testbed featuring additional components and functionalities such as chemical dosing systems, booster pumps and valves, as well as instrumentation and analyzers. It is collected over 16 days with \(m = 127\) dimensions.
- **Source:** [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad) is an industrial inspection dataset designed for benchmarking defects detection methods.
It consists of a 15 categories with a total of more than 5000 high-resolution (\(3 \times 1024 \times 1024\)) images.
Each category includes a set of defect-free training images and a test set containing images with different types of defects, as well as defect-free images. The dataset provides pixel-accurate ground truth annotations for the defect regions, which have been carefully annotated and reviewed by the authors to align with human interpretation of real-world defects.
- **Source:** [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) is a widely used reading comprehension dataset that includes 107,785 question-answer pairs based on 536 Wikipedia articles. The dataset was generated by crowdworkers who formulated questions and provided specific text segments or spans as answers. The answers have undergone rigorous crowdworkers selection, additional answer collection, and manual crosscheck processes, making them reliable ground truth explanations for the corresponding questions.


#### Collection Cadence
<!-- scope: telescope -->
<!-- info: Select **all applicable**: -->
**Static:** Data was collected once from single or multiple sources.


#### Data Integration
<!-- scope: periscope -->
<!-- info: List all fields collected from different sources, and specify if
they were included or excluded from the dataset.

Use additional notes to
capture any other relevant information or considerations.

(Usage Note: Duplicate and complete the following for each upstream
source.) -->
**Source**

**Included Fields**

Data fields of each datasets were collected and are included in the dataset. Each of them has high dimension (>50) which is not displayed here. Please refer to individual source for the detailed features.


#### Data Processing
<!-- scope: microscope -->
<!-- info: Summarize how data from different sources or methods aggregated,
processed, or connected.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the following for each source OR
collection method.) -->
**Collection Method or Source**

**Description:** In our data processing pipeline, we employ different techniques based on the data type. For timeseries data, we apply normalization to ensure it falls within the range of [0, 1], enabling better comparison and analysis across different variables. On the other hand, we do not perform any additional processing for image and text data, as they are inherently suitable for analysis without preprocessing steps.

When it comes to annotations, we have a dedicated process to handle them. For ground truth annotation files, which are typically stored in formats such as Excel or PDF, we extract the relevant information such as start time, end time, and the sensors involved in the attack. We then align this information with the raw data to ensure accurate labeling of explanations. This process allows us to establish a clear link between the annotated events and the underlying data, facilitating the evaluation and analysis of the explanations provided by our models.

By leveraging these data processing techniques, we ensure that the data is appropriately prepared and annotated for further analysis and evaluation. This enables us to derive valuable insights and make informed decisions based on the processed and labeled data.

**Methods employed:** Normalization.

**Tools or libraries:** [Min-Max scaling](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).



### Collection Criteria
#### Data Selection
<!-- scope: telescope -->
<!-- info: Summarize the data selection criteria.

Use additional notes to capture any other relevant information or
considerations. -->
- **Collection Method of Source:** We select the dataset based on availability of ground truth of explanations.



#### Data Inclusion
<!-- scope: periscope -->
<!-- info: Summarize the data inclusion criteria.

Use additional notes to capture any other relevant information or
considerations. -->
- **Collection Method of Source:** Same as above.


#### Data Exclusion
<!-- scope: microscope -->
<!-- info: Summarize the data exclusion criteria.

Use additional notes to capture any other relevant information or
considerations. -->
- **Collection Method of Source:** We exclude data that does not have ground truth for explanation.


### Relationship to Source
#### Use & Utility(ies)
<!-- scope: telescope -->
<!-- info: Describe how the resulting dataset is aligned with the purposes,
motivations, or intended use of the upstream source(s).

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each source type.) -->
- **Source Type:** Summarize here. Include links where available.
- **Source Type:** Summarize here. Include links where available.
- **Source Type:** Summarize here. Include links where available.

**Additional Notes:** Add here

#### Benefit and Value(s)
<!-- scope: periscope -->
<!-- info: Summarize the benefits of the resulting dataset to its consumers,
compared to the upstream source(s).

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each source type.) -->
- **Source Type:** Summarize here. Include links where available.
- **Source Type:** Summarize here. Include links where available.
- **Source Type:** Summarize here. Include links where available.

**Additional Notes:** Add here

#### Limitation(s) and Trade-Off(s)
<!-- scope: microscope -->
<!-- info: What are the limitations of the resulting dataset to its consumers,
compared to the upstream source(s)?

Break down by source type.<br><br>(Usage Note: Duplicate and complete the
following for each source type.) -->
- **Source Type:** Summarize here. Include links where available.
- **Source Type:** Summarize here. Include links where available.
- **Source Type:** Summarize here. Include links where available.

### Version and Maintenance
<!-- info: Fill this next row if this is not the first version of the dataset,
and there is no data card available for the first version -->
#### First Version
<!-- scope: periscope -->
<!-- info: Provide a **basic description of the first version** of this
dataset. -->
- **Release date:** MM/YYYY
- **Link to dataset:** [Dataset Name + Version]
- **Status:** [Select one: Actively Maintained/Limited Maintenance/Deprecated]
- **Size of Dataset:** 123 MB
- **Number of Instances:** 123456

#### Note(s) and Caveat(s)
<!-- scope: microscope -->
<!-- info: Summarize the caveats or nuances of the first version of this
dataset that may affect the use of the current version.

Use additional notes to capture any other relevant information or
considerations. -->
Summarize here. Include links where available.

**Additional Notes:** Add here

#### Cadence
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Yearly
- Quarterly
- Monthly
- Biweekly
- Weekly
- Daily
- Hourly
- Static
- Others (please specify)

#### Last and Next Update(s)
<!-- scope: periscope -->
<!-- info: Please describe the update schedule: -->
- **Date of last update:** DD/MM/YYYY
- **Total data points affected:** 12345
- **Data points updated:** 12345
- **Data points added:** 12345
- **Data points removed:** 12345
- **Date of next update:** DD/MM/YYYY

#### Changes on Update(s)
<!-- scope: microscope -->
<!-- info: Summarize the changes that occur when the dataset is refreshed.

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each source type.) -->
- **Source Type:** Summarize here. Include links where available.
- **Source Type:** Summarize here. Include links where available.
- **Source Type:** Summarize here. Include links where available.

**Additional Notes:** Add here

## Human and Other Sensitive Attributes
#### Sensitive Human Attribute(s)
<!-- scope: telescope -->
<!-- info: Select **all attributes** that are represented (directly or
indirectly) in the dataset. -->
- Gender
- Socio-economic status
- Geography
- Language
- Age
- Culture
- Experience or Seniority
- Others (please specify)

#### Intentionality
<!-- scope: periscope -->
<!-- info: List fields in the dataset that contain human attributes, and
specify if their collection was intentional or unintentional.

Use additional notes to capture any other relevant information or
considerations. -->
**Intentionally Collected Attributes**

Human attributes were labeled or collected as a part of the dataset creation
process.

Field Name | Description
--- | ---
Field Name | Human Attributed Collected
Field Name | Human Attributed Collected

**Additional Notes:** Add here

**Unintentionally Collected Attributes**

Human attributes were not explicitly collected as a part of the dataset
creation process but can be inferred using additional methods.

Field Name | Description
--- | ---
Field Name | Human Attributed Collected
Field Name | Human Attributed Collected

**Additional Notes:** Add here

#### Rationale
<!-- scope: microscope -->
<!-- info: Describe the motivation, rationale, considerations or approaches
that caused this dataset to include the indicated human attributes.

Summarize why or how this might affect the use of the dataset. -->
Summarize here. Include links, table, and media as relevant.

#### Source(s)
<!-- scope: periscope -->
<!-- info: List the sources of the human attributes.

Use additional notes to capture any other relevant information or
considerations. -->
- **Human Attribute:** Sources
- **Human Attribute:** Sources
- **Human Attribute:** Sources

**Additional Notes:** Add here

#### Methodology Detail(s)
<!-- scope: microscope -->
<!-- info: Describe the methods used to collect human attributes in the
dataset.

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each human
attribute.) -->

**Human Attribute Method:** Describe the collection method here. Include links where necessary

**Collection task:** Describe the task here. Include links where necessary

**Platforms, tools, or libraries:**

- [Platform, tools, or libraries]: Write description here
- [Platform, tools, or libraries]: Write description here
- [Platform, tools, or libraries]: Write description here

**Additional Notes:** Add here

#### Distribution(s)
<!-- width: full -->
<!-- info: Provide basic descriptive statistics for each human attribute,
noting key takeaways in the caption.

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each human
attribute.) -->
Human Attribute | Label or Class | Label or Class | Label or Class | Label or Class
--- | --- | --- | --- | ---
Count | 123456 | 123456 | 123456 | 123456
[Statistic] | 123456 | 123456 | 123456 | 123456
[Statistic] | 123456 | 123456 | 123456 | 123456
[Statistic] | 123456 | 123456 | 123456 | 123456

**Above:** Provide a caption for the above table or visualization.
**Additional Notes:** Add here

#### Known Correlations
<!-- scope: periscope -->
<!-- info: Describe any known correlations with the indicated sensitive
attributes in this dataset.

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate for each known correlation.) -->
[`field_name`, `field_name`]

**Description:** Summarize here. Include visualizations, metrics, or links
where necessary.

**Impact on dataset use:** Summarize here. Include visualizations, metrics, or
links where necessary.

**Additional Notes:** add here

#### Risk(s) and Mitigation(s)
<!-- scope: microscope -->
<!-- info: Summarize systemic or residual risks, performance expectations,
trade-offs and caveats because of human attributes in this dataset.

Use additional notes to capture any other relevant information or
considerations.

Usage Note: Duplicate and complete the following for each human attribute. -->
**Human Attribute**

Summarize here. Include links and metrics where applicable.

**Risk type:** [Description + Mitigations]

**Risk type:** [Description + Mitigations]

**Risk type:** [Description + Mitigations]

**Trade-offs, caveats, & other considerations:** Summarize here. Include
visualizations, metrics, or links where necessary.

**Additional Notes:** Add here

## Extended Use
### Use with Other Data
#### Safety Level
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Safe to use with other data
- Conditionally safe to use with other data
- Should not be used with other data
- Unknown
- Others (please specify)

#### Known Safe Dataset(s) or Data Type(s)
<!-- scope: periscope -->
<!-- info: List the known datasets or data types and corresponding
transformations that **are safe to join or aggregate** this dataset with. -->
**Dataset or Data Type:** Summarize here. Include visualizations, metrics,
or links where necessary.

**Dataset or Data Type:** Summarize here. Include visualizations, metrics,
or links where necessary.

**Dataset or Data Type:** Summarize here. Include visualizations, metrics,
or links where necessary.

#### Best Practices
<!-- scope: microscope -->
<!-- info: Summarize best practices for using this dataset with other datasets
or data types.

Use additional notes to capture any other relevant information or
considerations. -->
Summarize here. Include visualizations, metrics, demonstrative examples,
or links where necessary.

**Additional Notes:** Add here

#### Known Unsafe Dataset(s) or Data Type(s)
<!-- scope: periscope -->
<!-- info: Fill this out if you selected "Conditionally safe to use with other
datasets" or "Should not be used with other datasets":

List the known datasets or data types and corresponding transformations that
are **unsafe to join or aggregate** with this dataset. -->
**Dataset or Data Type:** Summarize here. Include visualizations, metrics,
or links where necessary.

**Dataset or Data Type:** Summarize here. Include visualizations, metrics,
or links where necessary.

**Dataset or Data Type:** Summarize here. Include visualizations, metrics,
or links where necessary.

#### Limitation(s) and Recommendation(s)
<!-- scope: microscope -->
<!-- info: Fill this out if you selected "Conditionally safe to use with
other datasets" or "Should not be used with
other datasets":

Summarize limitations of the dataset that introduce foreseeable risks when the
dataset is conjoined with other datasets.

Use additional notes to capture any other relevant information or
considerations. -->
Summarize here. Include links and metrics where applicable.

**Limitation type:** Dataset or data type, description and recommendation.

**Limitation type:** Dataset or data type, description and recommendation.

**Limitation type:** Dataset or data type, description and recommendation.

**Additional Notes:** Add here

### Forking & Sampling
#### Safety Level
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Safe to form and/or sample
- Conditionally safe to fork and/or sample
- Should not be forked and/or sampled
- Unknown
- Others (please specify)

#### Acceptable Sampling Method(s)
<!-- scope: periscope -->
<!-- info: Select **all applicable** acceptable methods to sample this
dataset: -->
- Cluster Sampling
- Haphazard Sampling
- Multi-stage sampling
- Random Sampling
- Retrospective Sampling
- Stratified Sampling
- Systematic Sampling
- Weighted Sampling
- Unknown
- Unsampled
- Others (please specify)

#### Best Practice(s)
<!-- scope: microscope -->
<!-- info: Summarize the best practices for forking or sampling this dataset.

Use additional notes to capture any other relevant information or
considerations. -->
Summarize here. Include links, figures, and demonstrative examples where
available.

**Additional Notes:** Add here

#### Risk(s) and Mitigation(s)
<!-- scope: periscope -->
<!-- info: Fill this out if you selected "Conditionally safe to fork and/or
sample" or "Should not be forked and/or sampled":

Summarize known or residual risks associated with forking and sampling methods
when applied to the dataset.

Use additional notes to capture any other
relevant information or considerations. -->
Summarize here. Include links and metrics where applicable.

**Risk Type:** [Description + Mitigations]

**Risk Type:** [Description + Mitigations]

**Risk Type:** [Description + Mitigations]

**Additional Notes:** Add here

#### Limitation(s) and Recommendation(s)
<!-- scope: microscope -->
<!-- info: Fill this out if you selected "Conditionally safe to fork and/or
sample" or "Should not be forked and/or sample":

Summarize the limitations that the dataset introduces when forking
or sampling the dataset and corresponding recommendations.

Use additional notes to capture any other relevant information or
considerations. -->
Summarize here. Include links and metrics where applicable.

**Limitation Type:** [Description + Recommendation]

**Limitation Type:** [Description + Recommendation]

**Limitation Type:** [Description + Recommendation]

**Additional Notes:** Add here

### Use in ML or AI Systems
#### Dataset Use(s)
<!-- scope: telescope -->
<!-- info: Select **all applicable** -->
- Training
- Testing
- Validation
- Development or Production Use
- Fine Tuning
- Others (please specify)

#### Notable Feature(s)
<!-- scope: periscope -->
<!-- info: Describe any notable feature distributions or relationships between
individual instances made explicit.

Include links to servers where readers can explore the data on their own. -->

**Exploration Demo:** [Link to server or demo.]

**Notable Field Name:** Describe here. Include links, data examples, metrics,
visualizations where relevant.

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Usage Guideline(s)
<!-- scope: microscope -->
<!-- info: Summarize usage guidelines or policies that consumers should be
aware of.

Use additional notes to capture any other relevant information or
considerations. -->
**Usage Guidelines:** Summarize here. Include links where necessary.

**Approval Steps:** Summarize here. Include links where necessary.

**Reviewer:** Provide the name of a reviewer for publications referencing
this dataset.

**Additional Notes:** Add here

#### Distribution(s)
<!-- scope: periscope -->
<!-- info: Describe the recommended splits and corresponding criteria.

Use additional notes to capture any other
relevant information or considerations. -->

Set | Number of data points
--- | ---
Train | 62,563
Test | 62,563
Validation | 62,563
Dev | 62,563

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Known Correlation(s)
<!-- scope: microscope -->
<!-- info: Summarize any known correlations with
the indicated features in this dataset.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate for each known
correlation.) -->
`field_name`, `field_name`

**Description:** Summarize here. Include
visualizations, metrics, or links where
necessary.

**Impact on dataset use:** Summarize here.
Include visualizations, metrics, or links
where necessary.

**Risks from correlation:** Summarize here.
Include recommended mitigative steps if
available.

**Additional Notes:** Add here

#### Split Statistics
<!-- scope: periscope -->
<!-- width: full -->
<!-- info: Provide the sizes of each split. As appropriate, provide any
descriptive statistics for features. -->

Statistic | Train | Test | Valid | Dev
--- | --- | --- | --- | ---
Count | 123456 | 123456 | 123456 | 123456
Descriptive Statistic | 123456 | 123456 | 123456 | 123456
Descriptive Statistic | 123456 | 123456 | 123456 | 123456
Descriptive Statistic | 123456 | 123456 | 123456 | 123456

**Above:** Caption for table above.

## Transformations
<!-- info: Fill this section if any transformations were applied in the
creation of your dataset. -->
### Synopsis
#### Transformation(s) Applied
<!-- scope: telescope -->
<!-- info: Select **all applicable** transformations
that were applied to the dataset. -->
- Anomaly Detection
- Cleaning Mismatched Values
- Cleaning Missing Values
- Converting Data Types
- Data Aggregation
- Dimensionality Reduction
- Joining Input Sources
- Redaction or Anonymization
- Others (Please specify)

#### Field(s) Transformed
<!-- scope: periscope -->
<!-- info: Provide the fields in the dataset that
were transformed.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete
the following for each transformation
type applied. Include the data types to
which fields were transformed.) -->
**Transformation Type**

Field Name | Source & Target
--- | ---
Field Name | Source Field: Target Field
Field Name | Source Field: Target Field
... | ...

**Additional Notes:** Add here

#### Library(ies) and Method(s) Used
<!-- scope: microscope -->
<!-- info: Provide a description of the methods
used to transform or process the
dataset.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete
the following for each transformation
type applied.) -->
**Transformation Type**

**Method:** Describe the transformation
method here. Include links where
necessary.

**Platforms, tools, or libraries:**
- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here

**Transformation Results:** Provide
results, outcomes, and actions taken
because of the transformations. Include
visualizations where available.

**Additional Notes:** Add here

### Breakdown of Transformations
<!-- info: Fill out relevant rows. -->
#### Cleaning Missing Value(s)
<!-- scope: telescope -->
<!-- info: Which fields in the data were missing
values? How many? -->
Summarize here. Include links where available.

**Field Name:** Count or description

**Field Name:** Count or description

**Field Name:** Count or description

#### Method(s) Used
<!-- scope: periscope -->
<!-- info: How were missing values cleaned?
What other choices were considered? -->
Summarize here. Include links where necessary.

**Platforms, tools, or libraries**

- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here

#### Comparative Summary
<!-- scope: microscope -->
<!-- info: Why were missing values cleaned using
this method (over others)? Provide
comparative charts showing before
and after missing values were cleaned. -->
Summarize here. Include links, tables, visualizations where available.

**Field Name** | **Diff**
--- | ---
Field Name | Before: After
Field Name | Before: After
... | ...

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Residual & Other Risk(s)
<!-- scope: telescope -->
<!-- info: What risks were introduced because of
this transformation? Which risks were
mitigated? -->
Summarize here. Include links and metrics where applicable.

- **Risk Type:** Description + Mitigations
- **Risk Type:** Description + Mitigations
- **Risk Type:** Description + Mitigations

#### Human Oversight Measure(s)
<!-- scope: periscope -->
<!-- info: What human oversight measures,
including additional testing,
investigations and approvals were
taken due to this transformation? -->
Summarize here. Include links where available.

#### Additional Considerations
<!-- scope: microscope -->
<!-- info: What additional considerations were
made? -->
Summarize here. Include links where available.

#### Cleaning Mismatched Value(s)
<!-- scope: telescope -->
<!-- info: Which fields in the data were corrected
for mismatched values? -->
Summarize here. Include links where available.

**Field Name:** Count or Description

**Field Name:** Count or Description

**Field Name:** Count or Description

#### Method(s) Used
<!-- scope: periscope -->
<!-- info: How were incorrect or mismatched
values cleaned? What other choices
were considered? -->
Summarize here. Include links where available.

#### Comparative Summary
<!-- scope: microscope -->
<!-- info: Why were incorrect or mismatched
values cleaned using this method (over
others)? Provide a comparative
analysis demonstrating before and
after values were cleaned. -->
Summarize here. Include links where available.

**Field Name** | **Diff**
--- | ---
Field Name | Before: After
Field Name | Before: After
... | ...

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Residual & Other Risk(s)
<!-- scope: telescope -->
<!-- info: What risks were introduced because of
this transformation? Which risks were
mitigated? -->
Summarize here. Include links and metrics where applicable.

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations

#### Human Oversight Measure(s)
<!-- scope: periscope -->
<!-- info: What human oversight measures,
including additional testing,
investigations and approvals were
taken due to this transformation? -->
Summarize here. Include links where available.

#### Additional Considerations
<!-- scope: microscope -->
<!-- info: What additional considerations were made? -->
Summarize here. Include links where available.

#### Anomalies
<!-- scope: telescope -->
<!-- info: How many anomalies or outliers were
detected?
If at all, how were detected anomalies
or outliers handled?
Why or why not? -->
Summarize here. Include links where available.

**Field Name:** Count or Description

**Field Name:** Count or Description

**Field Name:** Count or Description

#### Method(s) Used
<!-- scope: periscope -->
<!-- info: What methods were used to detect
anomalies or outliers? -->
Summarize here. Include links where necessary.

**Platforms, tools, or libraries**

- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here

#### Comparative Summary
<!-- scope: microscope -->
<!-- info: Provide a comparative analysis
demonstrating before and after
anomaly handling measures. -->
Summarize here. Include links, tables, visualizations where available.

**Field Name** | **Diff**
--- | ---
Field Name | Before: After
Field Name | Before: After
... | ...

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Residual & Other Risk(s)
<!-- scope: telescope -->
<!-- info: What risks were introduced because of
this transformation? Which risks were
mitigated? -->
Summarize here. Include links and metrics where applicable.

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations

#### Human Oversight Measure(s)
<!-- scope: periscope -->
<!-- info: What human oversight measures,
including additional testing,
investigations and approvals were
taken due to this transformation? -->
Summarize here. Include links where available.

#### Additional Considerations
<!-- scope: microscope -->
<!-- info: What additional considerations were made? -->
Summarize here. Include links where available.

#### Dimensionality Reduction
<!-- scope: telescope -->
<!-- info: How many original features were
collected and how many dimensions
were reduced? -->
Summarize here. Include links where available.

**Field Name:** Count or Description

**Field Name:** Count or Description

**Field Name:** Count or Description

#### Method(s) Used
<!-- scope: periscope -->
<!-- info: What methods were used to reduce the
dimensionality of the data? What other
choices were considered? -->
Summarize here. Include links where
necessary.

**Platforms, tools, or libraries**

- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here

#### Comparative Summary
<!-- scope: microscope -->
<!-- info: Why were features reduced using this
method (over others)? Provide
comparative charts showing before
and after dimensionality reduction
processes. -->
Summarize here. Include links, tables, visualizations where available.

**Field Name** | **Diff**
--- | ---
Field Name | Before: After
Field Name | Before: After
... | ...

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Residual & Other Risks
<!-- scope: telescope -->
<!-- info: What risks were introduced because of
this transformation? Which risks were
mitigated? -->
Summarize here. Include links and metrics where applicable.

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations

#### Human Oversight Measure(s)
<!-- scope: periscope -->
<!-- info: What human oversight measures,
including additional testing,
investigations and approvals were
taken due to this transformation? -->
Summarize here. Include links where available.

#### Additional Considerations
<!-- scope: microscope -->
<!-- info: What additional considerations were made? -->
Summarize here. Include links where available.

#### Joining Input Sources
<!-- scope: telescope -->
<!-- info: What were the distinct input sources that were joined? -->
Summarize here. Include links where available.

**Field Name:** Count or Description

**Field Name:** Count or Description

**Field Name:** Count or Description

#### Method(s) Used
<!-- scope: periscope -->
<!-- info: What are the shared columns of fields used to join these
sources? -->
Summarize here. Include links where necessary.

**Platforms, tools, or libraries**

- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here

#### Comparative Summary
<!-- scope: microscope -->
<!-- info: Why were features joined using this
method over others?

Provide comparative charts showing
before and after dimensionality
reduction processes. -->
Summarize here. Include links, tables, visualizations where available.

**Field Name** | **Diff**
--- | ---
Field Name | Before: After
Field Name | Before: After
... | ...

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Residual & Other Risk(s)
<!-- scope: telescope -->
<!-- info: What risks were introduced because of
this transformation? Which risks were
mitigated? -->
Summarize here. Include links and metrics where applicable.

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations

#### Human Oversight Measure(s)
<!-- scope: periscope -->
<!-- info: What human oversight measures,
including additional testing,
investigations and approvals were
taken due to this transformation? -->
Summarize here. Include links where
available.

#### Additional Considerations
<!-- scope: microscope -->
<!-- info: What additional considerations were
made? -->
Summarize here. Include links where
available.

#### Redaction or Anonymization
<!-- scope: telescope -->
<!-- info: Which features were redacted or
anonymized? -->
Summarize here. Include links where available.

**Field Name:** Count or Description

**Field Name:** Count or Description

**Field Name:** Count or Description

#### Method(s) Used
<!-- scope: periscope -->
<!-- info: What methods were used to redact or
anonymize data? -->
Summarize here. Include links where necessary.

**Platforms, tools, or libraries**

- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here

#### Comparative Summary
<!-- scope: microscope -->
<!-- info: Why was data redacted or anonymized
using this method over others? Provide
comparative charts showing before
and after redaction or anonymization
process. -->
Summarize here. Include links, tables, visualizations where available.

**Field Name** | **Diff**
--- | ---
Field Name | Before: After
Field Name | Before: After
... | ...

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Residual & Other Risk(s)
<!-- scope: telescope -->
<!-- info: What risks were introduced because of
this transformation? Which risks were
mitigated? -->
Summarize here. Include links and metrics where applicable.

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations

#### Human Oversight Measure(s)
<!-- scope: periscope -->
<!-- info: What human oversight measures,
including additional testing,
investigations and approvals were
taken due to this transformation? -->
Summarize here. Include links where available.

#### Additional Considerations
<!-- scope: microscope -->
<!-- info: What additional considerations were
made? -->
Summarize here. Include links where available.

#### Others (Please Specify)
<!-- scope: telescope -->
<!-- info: What was done? Which features or
fields were affected? -->
Summarize here. Include links where available.

**Field Name:** Count or Description

**Field Name:** Count or Description

**Field Name:** Count or Description

#### Method(s) Used
<!-- scope: periscope -->
<!-- info: What method were used? -->
Summarize here. Include links where necessary.

**Platforms, tools, or libraries**

- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here

#### Comparative Summary
<!-- scope: microscope -->
<!-- info: Why was this method used over
others? Provide comparative charts
showing before and after this
transformation. -->
Summarize here. Include links, tables, visualizations where available.

**Field Name** | **Diff**
--- | ---
Field Name | Before: After
Field Name | Before: After
... | ...

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Residual & Other Risk(s)
<!-- scope: telescope -->
<!-- info: What risks were introduced because of
this transformation? Which risks were
mitigated? -->
Summarize here. Include links and metrics where applicable.

**Risk type:** [Description + Mitigations]

**Risk type:** [Description + Mitigations]

**Risk type:** [Description + Mitigations]

#### Human Oversight Measure(s)
<!-- scope: periscope -->
<!-- info: What human oversight measures,
including additional testing,
investigations and approvals were
taken due to this transformation? -->
Summarize here. Include links where available.

#### Additional Considerations
<!-- scope: microscope -->
<!-- info: What additional considerations were made? -->
Summarize here. Include links where available.

## Annotations & Labeling
<!-- info: Fill this section if any human or algorithmic annotation tasks were
performed in the creation of your dataset. -->
#### Annotation Workforce Type
<!-- scope: telescope -->
<!-- info: Select **all applicable** annotation
workforce types or methods used
to annotate the dataset: -->
- Annotation Target in Data
- Machine-Generated
- Annotations
- Human Annotations (Expert)
- Human Annotations (Non-Expert)
- Human Annotations (Employees)
- Human Annotations (Contractors)
- Human Annotations (Crowdsourcing)
- Human Annotations (Outsourced / Managed)
- Teams
- Unlabeled
- Others (Please specify)

#### Annotation Characteristic(s)
<!-- scope: periscope -->
<!-- info: Describe relevant characteristics of annotations
as indicated. For quality metrics, consider
including accuracy, consensus accuracy, IRR,
XRR at the appropriate granularity (e.g. across
dataset, by annotator, by annotation, etc.).

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each annotation type.) -->
**Annotation Type** | **Number**
--- | ---
Number of unique annotations | 123456789
Total number of annotations | 123456789
Average annotations per example | 123456789
Number of annotators per example | 123456789
[Quality metric per granuality] | 123456789
[Quality metric per granuality] | 123456789
[Quality metric per granuality] | 123456789

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Annotation Description(s)
<!-- scope: microscope -->
<!-- info: Provide descriptions of the annotations
applied to the dataset. Include links
and indicate platforms, tools or libraries
used wherever possible.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete
the following for each annotation
type.) -->
**(Annotation Type)**

**Description:** Description of annotations (labels, ratings) produced.
Include how this was created or authored.

**Link:** Relevant URL link.

**Platforms, tools, or libraries:**

- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here

**Additional Notes:** Add here

#### Annotation Distribution(s)
<!-- scope: periscope -->
<!-- info: Provide a distribution of annotations for each
annotation or class of annotations using the
format below.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each annotation type.) -->
**Annotation Type** | **Number**
--- | ---
Annotations (or Class) | 12345 (20%)
Annotations (or Class) | 12345 (20%)
Annotations (or Class) | 12345 (20%)
Annotations (or Class) | 12345 (20%)
Annotations (or Class) | 12345 (20%)

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Annotation Task(s)
<!-- scope: microscope -->
<!-- info: Summarize each task type associated
with annotations in the dataset.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete
the following for each task type.) -->
**(Task Type)**

**Task description:** Summarize here. Include links if available.

**Task instructions:** Summarize here. Include links if available.

**Methods used:** Summarize here. Include links if available.

**Inter-rater adjudication policy:** Summarize here. Include links if
available.

**Golden questions:** Summarize here. Include links if available.

**Additional notes:** Add here

### Human Annotators
<!-- info: Fill this section if human annotators were used. -->
#### Annotator Description(s)
<!-- scope: periscope -->
<!-- info: Provide a brief description for each annotator
pool performing the human annotation task.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each annotation type.) -->
**(Annotation Type)**

**Task type:** Summarize here. Include links if available.

**Number of unique annotators:** Summarize here. Include links if available.

**Expertise of annotators:** Summarize here. Include links if available.

**Description of annotators:** Summarize here. Include links if available.

**Language distribution of annotators:** Summarize here. Include links if
available.

**Geographic distribution of annotators:** Summarize here. Include links if
available.

**Summary of annotation instructions:** Summarize here. Include links if
available.

**Summary of gold questions:** Summarize here. Include links if available.

**Annotation platforms:** Summarize here. Include links if available.

**Additional Notes:** Add here

#### Annotator Task(s)
<!-- scope: microscope -->
<!-- info: Provide a brief description for each
annotator pool performing the human
annotation task.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete
the following for each annotation
type.) -->
**(Task Type)**

**Task description:** Summarize here. Include links if available.

**Task instructions:** Summarize here. Include links if available.

**Methods used:** Summarize here. Include links if available.

**Inter-rater adjudication policy:** Summarize here. Include links if
available.

**Golden questions:** Summarize here. Include links if available.

**Additional notes:** Add here

#### Language(s)
<!-- scope: telescope -->
<!-- info: Provide annotator distributions for
each annotation type.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and
complete the following for each
annotation type.) -->
**(Annotation Type)**

- Language [Percentage %]
- Language [Percentage %]
- Language [Percentage %]

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Location(s)
<!-- scope: periscope -->
<!-- info: Provide annotator distributions for each
annotation type.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each annotation type.) -->
**(Annotation Type)**

- Location [Percentage %]
- Location [Percentage %]
- Location [Percentage %]

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Gender(s)
<!-- scope: microscope -->
<!-- info: Provide annotator distributions for
each annotation type.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete
the following for each annotation
type.) -->
**(Annotation Type)**

- Gender [Percentage %]
- Gender [Percentage %]
- Gender [Percentage %]

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

## Validation Types
<!-- info: Fill this section if the data in the dataset was validated during
or after the creation of your dataset. -->
#### Method(s)
<!-- scope: telescope -->
<!-- info: Select **all applicable**: -->
- Data Type Validation
- Range and Constraint Validation
- Code/cross-reference Validation
- Structured Validation
- Consistency Validation
- Not Validated
- Others (Please Specify)

#### Breakdown(s)
<!-- scope: periscope -->
<!-- info: Provide a description of the fields and data
points that were validated.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each validator type.) -->
**(Validation Type)**

**Number of Data Points Validated:** 12345

**Fields Validated**
Field | Count (if available)
--- | ---
Field | 123456
Field | 123456
Field | 123456

**Above:** Provide a caption for the above table or visualization.

#### Description(s)
<!-- scope: microscope -->
<!-- info: Provide a description of the methods used to
validate the dataset.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each validator type.) -->
**(Validation Type)**

**Method:** Describe the validation method here. Include links where
necessary.

**Platforms, tools, or libraries:**

- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here

**Validation Results:** Provide results, outcomes, and actions taken because
of the validation. Include visualizations where available.

**Additional Notes:** Add here

### Description of Human Validators
<!-- info: Fill this section if the dataset was validated using human
validators -->
#### Characteristic(s)
<!-- scope: periscope -->
<!-- info: Provide characteristics of the validator
pool(s). Use additional notes to capture any
other relevant information or considerations. -->
**(Validation Type)**
- Unique validators: 12345
- Number of examples per validator: 123456
- Average cost/task/validator: $$$
- Training provided: Y/N
- Expertise required: Y/N

#### Description(s)
<!-- scope: microscope -->
<!-- info: Provide a brief description of the validator
pool(s). Use additional notes to capture any
other relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each validator type.) -->
**(Validation Type)**

**Validator description:** Summarize here. Include links if available.

**Training provided:** Summarize here. Include links if available.

**Validator selection criteria:** Summarize here. Include links if available.

**Training provided:** Summarize here. Include links if available.

**Additional Notes:** Add here

#### Language(s)
<!-- scope: telescope -->
<!-- info: Provide validator distributions.
Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each annotation type.)-->
**(Validation Type)**

- Language [Percentage %]
- Language [Percentage %]
- Language [Percentage %]

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Location(s)
<!-- scope: periscope -->
<!-- info: Provide validator distributions.
Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each annotation type.)-->
**(Validation Type)**

- Location [Percentage %]
- Location [Percentage %]
- Location [Percentage %]

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Gender(s)
<!-- scope: microscope -->
<!-- info: Provide validator distributions.
Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each annotation type.)-->
**(Validation Type)**

- Gender [Percentage %]
- Gender [Percentage %]
- Gender [Percentage %]

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

## Sampling Methods
<!-- info: Fill out the following block if your dataset employs any sampling
methods. -->
#### Method(s) Used
<!-- scope: telescope -->
<!-- info: Select **all applicable** methods used in the creation of this
dataset: -->
- Cluster Sampling
- Haphazard Sampling
- Multi-stage Sampling
- Random Sampling
- Retrospective Sampling
- Stratified Sampling
- Systematic Sampling
- Weighted Sampling
- Unknown
- Unsampled
- Others (Please specify)

#### Characteristic(s)
<!-- scope: periscope -->
<!-- info: Provide characteristics of each sampling
method used.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each sampling method
used.) -->
**(Sampling Type)** | **Number**
--- | ---
Upstream Source | Write here
Total data sampled | 123m
Sample size | 123
Threshold applied | 123k units at property
Sampling rate | 123
Sample mean | 123
Sample std. dev | 123
Sampling distribution | 123
Sampling variation | 123
Sample statistic | 123

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Sampling Criteria
<!-- scope: microscope -->
<!-- info: Describe the criteria used to sample data from
upstream sources.

Use additional notes to capture any other
relevant information or considerations. -->
- **Sampling method:** Summarize here. Include links where applicable.
- **Sampling method:** Summarize here. Include links where applicable.
- **Sampling method:** Summarize here. Include links where applicable.

## Known Applications & Benchmarks
<!-- info: Fill out the following section if your dataset was primarily
created for use in AI or ML system(s) -->
#### ML Application(s)
<!-- scope: telescope -->
<!-- info: Provide a list of key ML tasks
that the dataset has been
used for.

Usage Note: Use comma-separated keywords. -->
*For example: Classification, Regression, Object Detection*

#### Evaluation Result(s)
<!-- scope: periscope -->
<!-- info: Provide the evaluation results from
models that this dataset has been used
in.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete the
following for each model.) -->
**(Model Name)**

**Model Card:** [Link to full model card]

Evaluation Results

- Accuracy: 123 (params)
- Precision: 123 (params)
- Recall: 123 (params)
- Performance metric: 123 (params)

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Evaluation Process(es)
<!-- scope: microscope -->
<!-- info: Provide a description of the evaluation process for
the model's overall performance or the
determination of how the dataset contributes to
the model's performance.

Use additional notes to capture any other relevant
information or considerations.

(Usage Note: Duplicate and complete the following
for each model and method used.) -->
**(Model Name)**

**[Method used]:** Summarize here. Include links where available.

- **Process:** Summarize here. Include links, diagrams, visualizations, tables as relevant.
- **Factors:** Summarize here. Include links, diagrams, visualizations, tables as relevant.
- **Considerations:** Summarize here. Include links, diagrams, visualizations, tables as relevant.
- **Results:** Summarize here. Include links, diagrams, visualizations, tables as relevant.

**Additional Notes:** Add here

#### Description(s) and Statistic(s)
<!-- scope: periscope -->
<!-- info: Provide a description of the model(s) and
task(s) that this dataset has been used
in.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete the
following for each model.) -->
**(Model Name)**

**Model Card:** Link to full model card

**Model Description:** Summarize here. Include links where applicable.

- Model Size: 123 (params)
- Model Weights: 123 (params)
- Model Layers 123 (params)
- Latency: 123 (params)

**Additional Notes:** Add here

#### Expected Performance and Known Caveats
<!-- scope: microscope -->
<!-- info: Provide a description of the expected performance
and known caveats of the models for this dataset.

Use additional notes to capture any other relevant
information or considerations.

(Usage Note: Duplicate and complete the following
for each model.) -->
**(Model Name)**

**Expected Performance:** Summarize here. Include links where available.

**Known Caveats:** Summarize here. Include links, diagrams, visualizations, and
tables as relevant.

**Additioanl Notes:** Add here

## Terms of Art
### Concepts and Definitions referenced in this Data Card
<!-- info: Use this space to include the expansions and definitions of any
acronyms, concepts, or terms of art used across the Data Card.
Use standard definitions where possible. Include the source of the definition
where indicated. If you are using an interpretation,
adaptation, or modification of the standard definition for the purposes of your
Data Card or dataset, include your interpretation as well. -->
#### Term of Art
Definition: Write here

Source: Write here and share link

Interpretation: Write here



