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
In our maintenance plan, our primary focus will be on preserving and leveraging the existing data that we have collected. This involves ensuring the integrity and security of the data through regular backups, implementing robust data storage practices, and conducting periodic audits to identify any potential issues or anomalies. Additionally, we recognize the growing importance of graph datasets in various domains. To capitalize on this, we will actively explore and evaluate potential graph datasets that align with our needs and objectives. This includes seeking out reliable sources, assessing the quality and relevance of the data, and integrating suitable graph datasets into our existing infrastructure. By incorporating graph datasets, we aim to enhance the depth and breadth of our analysis, uncover hidden patterns and relationships, and gain valuable insights that can drive informed decision-making and optimize our operations. In addition, we are aware that the SQuAD dataset does not have a clear classification task and may not align well with the remaining dataset. We are also exploring the Contract Understanding Atticus Dataset [(CUAD)](https://www.atticusprojectai.org/cuad) to see if we can algin the document classification task with the ground truth explanation they provide.

Our maintenance plan thus
combines the preservation of existing data with the exploration of new
graph and text datasets, ensuring a comprehensive and forward-looking approach to
data management and utilization.

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
**Updates to Data:** Next version of the dataset will possibly include suitable graph dataset.  We are currently investigating at the CUAD dataset.


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
- [Demo Link](https://github.com/xjiae/HDDDS/blob/main/example.ipynb)

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
- Direct download URL: [link](https://drive.google.com/uc?id=1RNF3NcfapWQYmz8OdxoivBTlqflhtrIj&export=download-anyway&confirm=t)


Code to download data:
```
wget -O data.zip "https://drive.google.com/uc?id=1RNF3NcfapWQYmz8OdxoivBTlqflhtrIj&export=download-anyway&confirm=t"
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

Data fields of each datasets were collected and are included in the dataset. Each of them has high dimension (>50), we found the detailed description for HAI and SWaT and consolidate them to the tables below. For WADI we did not find any detailed description. It is an extension of SWaT, and we attach the testbed information [here](https://itrust.sutd.edu.sg/itrust-labs-home/itrust-labs_wadi/).

HAI Features:
| Name         | Min Value|Max Value| Unit   | Description |
| ------------ | ------ | ------ | ----------- |
| P1_B2004     | 0      | 10     | bar         | Heat-exchanger outlet pressure setpoint |
| P1_B2016     | 0      | 10     | bar         | Pressure demand for thermal power output control |
| P1_B3004     | 0      | 720    | mm          | Water level setpoint (return water tank) |
| P1_B3005     | 0      | 2500   | l/h         | Discharge flowrate setpoint (return water tank) |
| P1_B4002     | 0      | 100    | ℃           | Heat-exchanger outlet temperature setpoint |
| P1_B4005     | 0      | 100    | %           | Temperature PID control output |
| P1_B400B     | 0      | 2500   | l/h         | Water outflow rate setpoint (heating water tank) |
| P1_B4022     | 0      | 40     | ℃           | Temperature demand for thermal power output control |
| P1_FCV01D    | 0      | 100    | %           | Position command for the FCV01 valve |
| P1_FCV01Z    | 0      | 100    | %           | Current position of the FCV01 valve |
| P1_FCV02D    | 0      | 100    | %           | Position command for the FCV02 valve |
| P1_FCV02Z    | 0      | 100    | %           | Current position of the FCV02 valve |
| P1_FCV03D    | 0      | 100    | %           | Position command for the FCV03 valve |
| P1_FCV03Z    | 0      | 100    | %           | Current position of the FCV03 valve |
| P1_FT01      | 0      | 2500   | mmH2O       | Measured flowrate of the return water tank |
| P1_FT01Z     | 0      | 3190   | l/h         | Water inflow rate converted from P1_FT01 |
| P1_FT02      | 0      | 2500   | mmH2O       | Measured flowrate of heating water tank |
| P1_FT02Z     | 0      | 3190   | l/h         | Water outflow rate conversion from P1_FT02 |
| P1_FT03      | 0      | 2500   | mmH2O       | Measured flowrate of the return water tank |
| P1_FT03Z     | 0      | 3190   | l/h         | Water outflow rate converted from P1_FT03 |
| P1_LCV01D    | 0      | 100    | %           | Position command for the LCV01 valve |
| P1_LCV01Z    | 0      | 100    | %           | Current position of the LCV01 valve |
| P1_LIT01     | 0      | 720    | mm          | Water level of the return water tank |
| P1_PCV01D    | 0      | 100    | %           | Position command for the PCV01 valve |
| P1_PCV01Z    | 0      | 100    | %           | Current position of the PCV01 valve |
| P1_PCV02D    | 0      | 100    | %           | Position command for the PCV2 valve |
| P1_PCV02Z    | 0      | 100    | %           | Current position of the PCV02 valve |
| P1_PIT01     | 0      | 10     | bar         | Heat-exchanger outlet pressure |
| P1_PIT01_HH  | 0      | 10     | bar         | Highest outlet pressure of the heat-exchanger |
| P1_PIT02     | 0      | 10     | bar         | Water supply pressure of the heating water pump |
| P1_PP01AD    | 0      | 1      | Boolean     | Start command of the main water pump PP01A |
| P1_PP01AR    | 0      | 1      | Boolean     | Running state of the main water pump PP01A |
| P1_PP01BD    | 0      | 1      | Boolean     | Start command of the main water pump PP01B |
| P1_PP01BR    | 0      | 1      | Boolean     | Running state of the main water pump PP01B |
| P1_PP02D     | 0      | 1      | Boolean     | Start command of the heating water pump PP02 |
| P1_PP02R     | 0      | 1      | Boolean     | Running state of the heating water pump PP02 |
| P1_PP04      | 0      | 100    | %           | Control out of the cooler pump |
| P1_PP04SP    | 0      | 100    | ℃           | Cooler temperature setpoint |
| P1_SOL01D    | 0      | 1      | Boolean     | Open command of the main water tank supply valve |
| P1_SOL03D    | 0      | 1      | Boolean     | Open command of the main water tank drain valve |
| P1_STSP      | 0      | 1      | Boolean     | Start/stop command of the boiler DCS |
| P1_TIT01     | \-50   | 150    | ℃           | Heat-exchanger outlet temperature |
| P1_TIT02     | \-50   | 150    | ℃           | Temperature of the heating water tank |
| P1_TIT03     | \-50   | 150    | ℃           | Temperature of the main water tank |
| P2_24Vdc     | 0      | 30     | Voltage     | DCS 24V Input Voltage |
| P2_ATSW_Lamp | 0      | 1      | Boolean     | Lamp of the Auto SW |
| P2_AutoGo    | 0      | 1      | Boolean     | Auto start button |
| P2_AutoSD    | 0      | 3200   | RPM         | Auto speed demand |
| P2_Emerg     | 0      | 1      | Boolean     | Emergency button |
| P2_MASW      | 0      | 1      | Boolean     | Manual(1)/Auto(0) SW |
| P2_MASW_Lamp | 0      | 1      | Boolean     | Lamp of Manual SW |
| P2_ManualGO  | 0      | 1      | Boolean     | Manual start button |
| P2_ManualSD  | 0      | 3200   | RPM         | Manual speed demand |
| P2_OnOff     | 0      | 1      | Boolean     | On/off switch of the turbine DCS |
| P2_RTR       | 0      | 2880   | RPM         | RPM trip rate |
| P2_SCO       | 0      | 100000 | \-          | Control output value of the speed controller |
| P2_SCST      | \-100  | 100    | RPM         | Speed change proportional to frequency change of the STM |
| P2_SIT01     | 0      | 3200   | RPM         | Current turbine RPM measured by speed probe |
| P2_TripEx    | 0      | 1      | Boolean     | Trip emergency exit button |
| P2_VIBTR01   | \-10   | 10     | ㎛           | Shaft-vibration-related Y-axis displacement near the 1st mass wheel |
| P2_VIBTR02   | \-10   | 10     | ㎛           | Shaft-vibration-related X-axis displacement near the 1st mass wheel |
| P2_VIBTR03   | \-10   | 10     | ㎛           | Shaft-vibration-related Y-axis displacement near the 2nd mass wheel |
| P2_VIBTR04   | \-10   | 10     | ㎛           | Shaft-vibration-related X-axis displacement near the 2nd mass wheel |
| P2_VT01      | 11     | 12     | rad/s       | Phase lag signal of the key phasor probe |
| P2_VTR01     | \-10   | 10     | ㎛           | Preset vibration limit for the sensor P2_VIBTR01 |
| P2_VTR02     | \-10   | 10     | ㎛           | Preset vibration limit for the sensor P2_VIBTR02 |
| P2_VTR03     | \-10   | 10     | ㎛           | Preset vibration limit for the sensor P2_VIBTR03 |
| P2_VTR04     | \-10   | 10     | ㎛           | Preset vibration limit for the sensor P2_VIBTR03 |
| P3_FIT01     | 0      | 27648  | \-          | Flow rate of water flowing into the upper water tank |
| P3_LCP01D    | 0      | 27648  | \-          | Speed command for the pump LCP01 |
| P3_LCV01D    | 0      | 27648  | \-          | Position command for the valve LCV01 |
| P3_LH01      | 0      | 70     | %           | High water level set-point |
| P3_LIT01     | 0      | 90     | %           | Water level of the upper water tank |
| P3_LL01      | 0      | 70     | %           | Low water level set-point |
| P3_PIT01     | 0      | 27648  | \-          | Pressure of water flowing into the upper water tank |
| P4_HT_FD     | \-0.02 | 0.02   | mHz         | Frequency deviation of HTM |
| P4_HT_PO     | 0      | 100    | MW          | Output power of HTM |
| P4_HT_PS     | 0      | 100    | MW          | Scheduled power demand of HTM |
| P4_LD        | 0      | 500    | MW          | Total electrical load demand |
| P4_ST_FD     | \-0.02 | 0.02   | Hz          | Frequency deviation of STM |
| P4_ST_GOV    | 0      | 27648  | \-          | Gate opening rate of STM |
| P4_ST_LD     | 0      | 500    | MW          | Electrical load demand of STM |
| P4_ST_PO     | 0      | 500    | MW          | Output power of STM |
| P4_ST_PS     | 0      | 500    | MW          | Scheduled power demand of STM |
| P4_ST_PT01   | 0      | 27648  | \-          | Digital value of steam pressure of STM |
| P4_ST_TT01   | 0      | 27648  | \-          | Digital value of steam temperature of STM |

For SWaT:
| Feature        | Type     | Description                                                                                |
| -------------- | -------- | ------------------------------------------------------------------------------------------ |
| FIT-101        | Sensor   | Flow meter; Measures inflow into raw water tank.                                           |
| LIT-101        | Sensor   | Level Transmitter; Raw water tank level.                                                   |
| MV-101         | Actuator | Motorized valve; Controls water flow to the raw water tank.                                |
| P-101          | Actuator | Pump; Pumps water from raw water tank to second stage.                                     |
| P-102 (backup) | Actuator | Pump; Pumps water from raw water tank to second stage.                                     |
| AIT-201        | Sensor   | Conductivity analyser; Measures NaCl level.                                                |
| AIT-202        | Sensor   | pH analyser; Measures HCl level.                                                           |
| AIT-203        | Sensor   | ORP analyser; Measures NaOCl level.                                                        |
| FIT-201        | Sensor   | Flow Transmitter; Control dosing pumps.                                                    |
| MV-201         | Actuator | Motorized valve; Controls water flow to the UF feed water tank.                            |
| P-201          | Actuator | Dosing pump; NaCl dosing pump.                                                             |
| P-202 (backup) | Actuator | Dosing pump; NaCl dosing pump.                                                             |
| P-203          | Actuator | Dosing pump; HCl dosing pump.                                                              |
| P-204 (backup) | Actuator | Dosing pump; HCl dosing pump.                                                              |
| P-205          | Actuator | Dosing pump; NaOCl dosing pump.                                                            |
| P-206 (backup) | Actuator | Dosing pump; NaOCl dosing pump.                                                            |
| DPIT-301       | Sensor   | Di↵erential pressure indicating transmitter; Controls the back-wash process.               |
| FIT-301        | Sensor   | Flow meter; Measures the flow of water in the UF stage.                                    |
| LIT-301        | Sensor   | Level Transmitter; UF feed water tank level.                                               |
| MV-301         | Actuator | Motorized Valve; Controls UF-Backwash process.                                             |
| MV-302         | Actuator | Motorized  Valve;  Controls  water  from  UF  process  to  De-Chlorination unit.           |
| MV-303         | Actuator | Motorized Valve; Controls UF-Backwash drain.                                               |
| MV-304         | Actuator | Motorized Valve; Controls UF drain.                                                        |
| P-301 (backup) | Actuator | UF feed Pump; Pumps water from UF feed water tank to RO feed water tank via UF filtration. |
| P-302          | Actuator | UF feed Pump; Pumps water from UF feed water tank to RO feed water tank via UF filtration. |
| AIT-401        | Sensor   | RO hardness meter of water.                                                                |
| AIT-402        | Sensor   | ORP meter; Controls the NaHSO3dosing(P203), NaOCl dosing (P205).                           |
| FIT-401        | Sensor   | Flow Transmitter ; Controls the UV dechlorinator.                                          |
| LIT-401        | Actuator | Level Transmitter; RO feed water tank level.                                               |
| P-401 (backup) | Actuator | Pump; Pumps water from RO feed tank to UV dechlorinator.                                   |
| P-402          | Actuator | Pump; Pumps water from RO feed tank to UV dechlorinator.                                   |
| P-403          | Actuator | Sodium bi-sulphate pump.                                                                   |
| P-404 (backup) | Actuator | Sodium bi-sulphate pump.                                                                   |
| UV-401         | Actuator | Dechlorinator; Removes chlorine from water.                                                |
| AIT-501        | Sensor   | RO pH analyser; Measures HCl level.                                                        |
| AIT-502        | Sensor   | RO feed ORP analyser; Measures NaOCl level.                                                |
| AIT-503        | Sensor   | RO feed conductivity analyser; Measures NaCl level.                                        |
| AIT-504        | Sensor   | RO permeate conductivity analyser; Measures NaCl level.                                    |
| FIT-501        | Sensor   | Flow meter; RO membrane inlet flow meter.                                                  |
| FIT-502        | Sensor   | Flow meter; RO Permeate flow meter.                                                        |
| FIT-503        | Sensor   | Flow meter; RO Reject flow meter.                                                          |
| FIT-504        | Sensor   | Flow meter; RO re-circulation flow meter.                                                  |
| P-501          | Actuator | Pump; Pumps dechlorinated water to RO.                                                     |
| P-502 (backup) | Actuator | Pump; Pumps dechlorinated water to RO.                                                     |
| PIT-501        | Sensor   | Pressure meter; RO feed pressure.                                                          |
| PIT-502        | Sensor   | Pressure meter; RO permeate pressure.                                                      |
| PIT-503        | Sensor   | Pressure meter;RO reject pressure.                                                         |
| FIT-601        | Sensor   | Flow meter; UF Backwash flow meter.                                                        |
| P-601          | Actuator | Pump; Pumps water from RO permeate tank to raw water tank (not used for data collection).  |
| P-602          | Actuator | Pump; Pumps water from UF back wash tank to UF filter to clean the membrane.               |
| P-603          | Actuator | Not implemented in SWaT yet.                                                               |


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
- **Dataset:** The resulting Ground Truth eXplanation (GTX) dataset is closely aligned with the purposes, motivations, and intended use of the upstream sources (HAI, WADI, SWaT, MVTec, and SQuAD). Through meticulous cleaning and preprocessing of annotation files, the dataset provides accurate ground truth information for feature attribution evaluation in explainable AI. This alignment ensures that the GTX dataset is a valuable resource for benchmarking, model development, and educational purposes, enabling advancements in transparency, interpretability, and trustworthiness of AI systems across domains.
<!-- - **Source Type:** Summarize here. Include links where available.
- **Source Type:** Summarize here. Include links where available.

**Additional Notes:** Add here -->

#### Benefit and Value(s)
<!-- scope: periscope -->
<!-- info: Summarize the benefits of the resulting dataset to its consumers,
compared to the upstream source(s).

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each source type.) -->
- **Dataset:** The Ground Truth eXplanation (GTX) dataset provides consumers with curated and cleaned annotations, consolidating data from multiple sources. Compared to the upstream sources, it offers enhanced data quality, convenience, and relevance for evaluating and improving feature attribution methods in explainable AI.

<!-- **Additional Notes:** Add here -->

#### Limitation(s) and Trade-Off(s)
<!-- scope: microscope -->
<!-- info: What are the limitations of the resulting dataset to its consumers,
compared to the upstream source(s)?

Break down by source type.<br><br>(Usage Note: Duplicate and complete the
following for each source type.) -->
- **Dataset:** While the resulting Ground Truth eXplanation (GTX) dataset offers benefits, it also has certain limitations compared to the upstream sources. Firstly, the GTX dataset may have reduced granularity compared to the original upstream sources, as it involves cleaning and preprocessing steps that can result in some loss of detailed information. Secondly, the dataset's scope and coverage may be limited to specific features or attributes relevant to feature attribution evaluation, potentially excluding certain aspects present in the upstream sources. Additionally, the GTX dataset's generalizability may be constrained by the specific contexts and domains of the upstream sources, which may not fully represent the diverse range of applications and scenarios. It is important for consumers to consider these limitations and assess whether the available data adequately meets their specific needs and requirements.
### Version and Maintenance
<!-- info: Fill this next row if this is not the first version of the dataset,
and there is no data card available for the first version -->
#### First Version
<!-- scope: periscope -->
<!-- info: Provide a **basic description of the first version** of this
dataset. -->
- **Release date:** 06/2023
- **Link to dataset:** [GTX + 1.0](https://github.com/xjiae/HDDDS)
- **Status:** [Actively Maintained]
- **Size of Dataset:** 12 GB
- **Number of Instances:** 3,798,242

#### Note(s) and Caveat(s)
<!-- scope: microscope -->
<!-- info: Summarize the caveats or nuances of the first version of this
dataset that may affect the use of the current version.

Use additional notes to capture any other relevant information or
considerations. -->
We may update the dataset content if we find suitable graph dataset, but it will not affect the exitsing datasets.



#### Cadence
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Static


#### Last and Next Update(s)
<!-- scope: periscope -->
<!-- info: Please describe the update schedule: -->
- **Date of last update:** 14/06/2023
- **Total data points affected:** 3,798,242
- **Data points updated:** 3,798,242
- **Data points added:** 3,798,242
- **Data points removed:** 0
- **Date of next update:** 08/08/2023

#### Changes on Update(s)
<!-- scope: microscope -->
<!-- info: Summarize the changes that occur when the dataset is refreshed.

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each source type.) -->
- **Dataset:** Update five real-world datasets.
<!-- - **Source Type:** Summarize here. Include links where available.
- **Source Type:** Summarize here. Include links where available. -->

<!-- **Additional Notes:** Add here -->

## Extended Use
### Use with Other Data
#### Safety Level
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Safe to use with other data

#### Known Safe Dataset(s) or Data Type(s)
<!-- scope: periscope -->
<!-- info: List the known datasets or data types and corresponding
transformations that **are safe to join or aggregate** this dataset with. -->
**Data Type:** time-series, image, and text.

#### Best Practices
<!-- scope: microscope -->
<!-- info: Summarize best practices for using this dataset with other datasets
or data types.

Use additional notes to capture any other relevant information or
considerations. -->
When using the Ground Truth eXplanation (GTX) dataset with other datasets or data types, it is important to ensure data compatibility, identify common features, validate and cross-reference the data, consider contextual relevance, document assumptions and limitations, and perform exploratory analysis for insights.


#### Known Unsafe Dataset(s) or Data Type(s)
<!-- scope: periscope -->
<!-- info: Fill this out if you selected "Conditionally safe to use with other
datasets" or "Should not be used with other datasets":

List the known datasets or data types and corresponding transformations that
are **unsafe to join or aggregate** with this dataset. -->
**N/A:**.


### Forking & Sampling
#### Safety Level
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Safe to form and/or sample

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

#### Best Practice(s)
<!-- scope: microscope -->
<!-- info: Summarize the best practices for forking or sampling this dataset.

Use additional notes to capture any other relevant information or
considerations. -->
When forking or sampling the GTX dataset, best practices include clearly defining sampling criteria, maintaining representative samples, documenting the sampling methodology, considering sample size and statistical power, and validating the sample.

#### Risk(s) and Mitigation(s)
<!-- scope: periscope -->
<!-- info: Fill this out if you selected "Conditionally safe to fork and/or
sample" or "Should not be forked and/or sampled":

Summarize known or residual risks associated with forking and sampling methods
when applied to the dataset.

Use additional notes to capture any other
relevant information or considerations. -->
No known risk for sampling.

### Use in ML or AI Systems
#### Dataset Use(s)
<!-- scope: telescope -->
<!-- info: Select **all applicable** -->
- Training
- Testing
- Validation
- Development or Production Use
- Fine Tuning

#### Notable Feature(s)
<!-- scope: periscope -->
<!-- info: Describe any notable feature distributions or relationships between
individual instances made explicit.

Include links to servers where readers can explore the data on their own. -->

The GTX dataset exhibits notable feature distributions and explicit relationships between individual instances. Through careful curation, the dataset captures diverse real-world data types such as time-series, image, and text, each with its distinct feature distributions. These distributions may reveal patterns, trends, or variations in the data, providing valuable insights into the characteristics of different instances. Additionally, explicit relationships between individual instances can be identified through the ground truth annotations, which establish causal connections between features and the corresponding labels. These relationships help to elucidate the impact and importance of specific features in explaining the ground truth, contributing to the evaluation and improvement of feature attribution methods in explainable AI. By leveraging the feature distributions and explicit relationships within the dataset, researchers, practitioners, and educators can gain a deeper understanding of the data and make informed decisions in their respective domains.

#### Usage Guideline(s)
<!-- scope: microscope -->
<!-- info: Summarize usage guidelines or policies that consumers should be
aware of.

Use additional notes to capture any other relevant information or
considerations. -->
**Usage Guidelines:** When using the GTX dataset, consumers should comply with licensing and terms of use, provide proper attribution and citation, aim for reproducibility and transparency, practice responsible and ethical use, and foster communication and collaboration within the community.

**Approval Steps:** N/A.

**Reviewer:** Provide the name of a reviewer for publications referencing
this dataset.


#### Distribution(s)
<!-- scope: periscope -->
<!-- info: Describe the recommended splits and corresponding criteria.

Use additional notes to capture any other
relevant information or considerations. -->

Set | Number of data points
--- | ---
Train | 70%
Test | 20%
Validation | 10%

**Splits:** Recommand splts.


#### Known Correlation(s)
<!-- scope: microscope -->
<!-- info: Summarize any known correlations with
the indicated features in this dataset.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate for each known
correlation.) -->
All the features are correlated with each other in a given instance. Hence, user should treat them as a complete data point when process them.



## Transformations
<!-- info: Fill this section if any transformations were applied in the
creation of your dataset. -->
### Synopsis
#### Transformation(s) Applied
<!-- scope: telescope -->
<!-- info: Select **all applicable** transformations
that were applied to the dataset. -->
- Cleaning Missing Values
- Normalization

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

All features in time-series dataset are preprocessed. But user can also specified "raw" for contents to get the original dataset.

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

**Method:** For timeseries data, we apply normalization to ensure it falls within the range of [0, 1], enabling better comparison and analysis across different variables. 

**Platforms, tools, or libraries:**
- Platform, tool, or library: [sklearn.preprocessing.MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).

**Transformation Results:** All time-series values falls within the range of [0, 1].



### Breakdown of Transformations
<!-- info: Fill out relevant rows. -->
#### Cleaning Missing Value(s)
<!-- scope: telescope -->
<!-- info: Which fields in the data were missing
values? How many? -->
We fill missing sensor values with mean of the corresponding column.


#### Method(s) Used
<!-- scope: periscope -->
<!-- info: How were missing values cleaned?
What other choices were considered? -->

To handle missing sensor values, we replace them with the mean value of the corresponding column.

**Platforms, tools, or libraries**

- Platform, tool, or library: [pandas.DataFrame.fillna](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html)


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
Total number of annotations | 1,321,591,510
Average annotations per example | 17,962


**Annotation summary** .


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


The annotations applied to the dataset were manually performed by the author. The author meticulously reviewed the annotation file, ensuring precise alignment of the start and end times of each attack/defect. They annotated the affected features, indicating the specific features impacted during each attack. The annotation process involved a thorough analysis and interpretation of the data to ensure accuracy and consistency. For non-attacked/defect instances, an all zeroes annotation is generated automatically. No specific platforms, tools, or libraries were mentioned in the provided information.

#### Annotation Distribution(s)
<!-- scope: periscope -->
<!-- info: Provide a distribution of annotations for each
annotation or class of annotations using the
format below.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each annotation type.) -->
There are two classes of annotations, 1 for explanatory feature and 0 otherwise. We report the ratio for class 1.
**Annotation Type** | **Number**
--- | ---
HAI, column-wise | 1,034,580 (1.17%)
SWaT, column-wise | 2,785,671 (2.10%)
WADI, column-wise | 652,018 (1.52%)
MVTec, pixel-wise | 1,317,011,456 (4.38%)
SQuAD, start-end position pair | 107,785 (3.10%)

**Annotation summary:** We summarize the explanatory feature count and ratio.




## Terms of Art
### Concepts and Definitions referenced in this Data Card
<!-- info: Use this space to include the expansions and definitions of any
acronyms, concepts, or terms of art used across the Data Card.
Use standard definitions where possible. Include the source of the definition
where indicated. If you are using an interpretation,
adaptation, or modification of the standard definition for the purposes of your
Data Card or dataset, include your interpretation as well. -->
#### Term of Art
Definition: feature attribution

Interpretation: Feature attributions indicate how much each feature in your model contributed to the predictions for each given instance.



