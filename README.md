# 📚 Awesome-Self-Supervised-Time-Series-Anomaly-Detection


Welcome to this repository, where we gather some of the most relevant studies in the field of self-supervised learning for time series anomaly detection. These contributions are reviewed in our paper, "Self-Supervised Learning in Time Series Anomaly Detection: Recent Advances and Open Challenges", which is currently under review in the ACM Computing Surveys Journal. We hope you find this information useful. Enjoy! 😊

## 📘 Theoretical Background

Self-Supervised Time Series Anomaly Detection is a specialized field that builds on two key areas: self-supervised learning and time series anomaly detection. To help you get up to speed, we've curated a collection of essential papers that provide the necessary theoretical foundation in these topics.

🔍 **Self-Supervised Learning**:


- [Self-supervised visual feature learning with deep neural networks: A survey (2019)](https://arxiv.org/pdf/1902.06162)
- [Self-supervised learning: Generative or contrastive (2020)](https://arxiv.org/pdf/2006.08218)
- [A survey on contrastive self-supervised learning (2020)](https://arxiv.org/pdf/2011.00362)
- [A survey on self-supervised learning: Algorithms, applications, and future trends (2023)](https://arxiv.org/pdf/2301.05712)
- [Self-Supervised Learning for Time Series Analysis: Taxonomy, Progress, and Prospects (2023)](https://arxiv.org/pdf/2306.10125)

🚨 **Anomaly Detection**: 

- [Anomaly detection: A survey (2009)](http://cucis.ece.northwestern.edu/projects/DMS/publications/AnomalyDetection.pdf)
- [Deep learning for anomaly detection: A survey (2019)](https://arxiv.org/pdf/1901.03407)
- [Deep Learning for Anomaly Detection: A Review (2020)](https://arxiv.org/pdf/2007.02500)
- [Analyzing rare event, anomaly, novelty and outlier detection terms under the supervised classification framework (2020)](https://bird.bcamath.org/bitstream/handle/20.500.11824/1011/AIR_Analyzing_plain.pdf;jsessionid=47DF8BA773E74D6981269A9792283F4E?sequence=1)

📈 **Time Series Anomaly Detection**:

- [Outlier detection for temporal data: A survey (2014)](https://www.microsoft.com/en-us/research/wp-content/uploads/2014/01/gupta14_tkde.pdf)
- [A review on outlier/anomaly detection in time series data (2020)](https://arxiv.org/pdf/2002.04236)
- [Anomaly detection in univariate time-series: A survey on the state-of-the-art (2020)](https://arxiv.org/pdf/2004.00433)
- [Deep Learning for Time Series Anomaly Detection: A Survey (2022)](https://arxiv.org/pdf/2211.05244)
- [Anomaly detection in time series: a comprehensive evaluation (2022)](https://www.vldb.org/pvldb/vol15/p1779-wenig.pdf)


## 📍 Local Anomaly Detection in Time Series

Local anomaly detection in time series aims to identify anomalies that occur at specific points or small segments within an individual time series. These anomalies typically represent minor but significant deviations from the expected behavior, such as sudden spikes or drops. The following papers delve into self-supervised methods and strategies for detecting local anomalies in time series data.

### Self-Predictive Methods

- Anomaly detection using autoencoders with nonlinear dimensionality reduction (2014)  [[pdf]](https://dl.acm.org/doi/10.1145/2689746.2689747)
- A novel approach for automatic acoustic novelty detection using a denoising autoencoder with bidirectional LSTM neural networks (2015) [[pdf]](https://ieeexplore.ieee.org/document/7178320)
- Wind turbine fault detection using a denoising autoencoder with temporal information (2017) [[pdf]](https://ieeexplore.ieee.org/abstract/document/8059861)
- DeepAnT: A deep learning approach for unsupervised anomaly detection in time series (2018) [[pdf]](https://ieeexplore.ieee.org/document/8581424) [[code]](https://github.com/datacubeR/DeepAnt)
- A deep neural network for unsupervised anomaly detection and diagnosis in multivariate time series data (2019) [[pdf]](https://arxiv.org/abs/1811.08055) [[code]](https://github.com/wxdang/MSCRED)
- Spacecraft anomaly detection and relation visualization via masked time series modeling (2019) [[pdf]](https://ieeexplore.ieee.org/document/8943031)
- USAD: Unsupervised anomaly detection on multivariate time series (2020) [[pdf]](https://dl.acm.org/doi/10.1145/3394486.3403392) [[code]](https://github.com/manigalati/usad)
- Anomaly detection for wind turbines based on the reconstruction of condition parameters using stacked denoising autoencoders (2020) [[pdf]](https://www.sciencedirect.com/science/article/pii/S0960148119313710)
- Timeseries anomaly detection using temporal hierarchical one-class network (2020) [[pdf]](https://proceedings.neurips.cc/paper/2020/file/97e401a02082021fd24957f852e0e475-Paper.pdf)
- DeepFIB: Self-imputation for time series anomaly detection (2021) [[pdf]](https://arxiv.org/abs/2112.06247)
- Mad: Self-supervised masked anomaly detection task for multivariate time series (2022) [[pdf]](https://arxiv.org/abs/2205.02100)
- DUMA: Dual Mask for Multivariate Time Series Anomaly Detection (2022) [[pdf]](https://ieeexplore.ieee.org/document/9969633)
- Anomaly-PTG: a time series data-anomaly-detection transformer framework in multiple scenarios (2022) [[pdf]](https://www.mdpi.com/2079-9292/11/23/3955)
- Efficient time series anomaly detection by multiresolution self-supervised discriminative network (2022) [[pdf]](https://www.sciencedirect.com/science/article/pii/S0925231222003435)
- Self-Supervised Learning for Time-Series Anomaly Detection in Industrial Internet of Things (2022) [[pdf]](https://www.mdpi.com/2079-9292/11/14/2146)
- An Unsupervised Short-and Long-Term Mask Representation for Multivariate Time Series Anomaly Detection (2022) [[pdf]](https://arxiv.org/abs/2208.09240) [[code]](https://github.com/qiumiao30/SLMR)
- MAD-SGCN: Multivariate Anomaly Detection with Self-learning Graph Convolutional Networks (2022) [[pdf]](https://ieeexplore.ieee.org/document/9835470)
- AnomalyBERT: Self-Supervised Transformer for Time Series Anomaly Detection using Data Degradation Scheme (2023) [[pdf]](https://arxiv.org/abs/2305.04468) [[code]](https://github.com/Jhryu30/AnomalyBERT)

### Contrastive Methods

- Neural contextual anomaly detection for time series (2021) [[pdf]](https://arxiv.org/abs/2107.07702)
- Detecting anomalies within time series using local neural transformations (2022) [[pdf]](https://arxiv.org/abs/2202.03944) [[code]](https://github.com/boschresearch/local_neural_transformations)
- Ts2vec: Towards universal representation of time series (2022) [[pdf]](https://arxiv.org/abs/2106.10466) [[code]](https://github.com/yuezhihan/TS2Vec)
- Contrastive predictive coding for anomaly detection in multi-variate time series data (2022) [[pdf]](https://arxiv.org/abs/2202.03639)
- Stochastic pairing for contrastive anomaly detection on time series (2022) [[pdf]](https://laureberti.github.io/website/pub/icprai.pdf)
- DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection (2023) [[pdf]](https://arxiv.org/abs/2306.10347) [[code]](https://github.com/DAMO-DI-ML/KDD2023-DCdetector)
- Contrastive Time Series Anomaly Detection by Temporal Transformations (2023) [[pdf]](https://ieeexplore.ieee.org/document/10191358) [[code]](https://github.com/KDD-OpenSource/ContrastAD)
- Learning Robust and Consistent Time Series Representations: A Dilated Inception-Based Approach (2023) [[pdf]](https://arxiv.org/abs/2306.06579) [[code]](https://github.com/anhduy0911/CoInception)
- Time-series Anomaly Detection via Contextual Discriminative Contrastive Learning (2023) [[pdf]](https://arxiv.org/pdf/2304.07898)
- Unsafe Behavior Detection with Adaptive Contrastive Learning in Industrial Control Systems (2023) [[pdf]](https://ieeexplore.ieee.org/document/10190657)
- TiCTok: Time-Series Anomaly Detection with Contrastive Tokenization (2023) [[pdf]](https://www.researchgate.net/publication/372871589_TiCTok_Time-Series_Anomaly_Detection_with_Contrastive_Tokenization)

### Self-Predictive + Contrastive Methods

- Time Series Anomaly Detection using Skip-Step Contrastive Predictive Coding (2022) [[pdf]](https://sslneurips22.github.io/paper_pdfs/paper_19.pdf)
- Unsupervised time-series representation learning with iterative bilinear temporal-spectral fusion (2022) [[pdf]](https://arxiv.org/abs/2202.04770)
- Warping resilient scalable anomaly detection in time series (2022) [[pdf]](https://arxiv.org/abs/1906.05205) [[code]](https://github.com/WaRTEm-AD/UnivariateAnomalydetection)
- Contrastive autoencoder for anomaly detection in multivariate time series (2022) [[pdf]](https://www.sciencedirect.com/science/article/pii/S0020025522008775)
- Time Series Anomaly Detection for Smart Grids via Multiple Self-Supervised Tasks Learning (2022) [[pdf]](https://ieeexplore.ieee.org/document/10030049)
- Calibrated one-class classification for unsupervised time series anomaly detection (2022) [[pdf]](https://arxiv.org/abs/2207.12201) [[code]](https://github.com/xuhongzuo/couta)
- Multi-Task Self-Supervised Time-Series Representation Learning (2023) [[pdf]](https://arxiv.org/abs/2303.01034)

## 🌏 Global Anomaly Detection in Time Series

Global anomaly detection in time series refers to the identification of entire time series that act as outliers in a dataset comprising numerous sequences. The methods employed in this scenario aim to uncover global patterns that characterize the sequences across the entire dataset at the sample level. The following papers provide a comprehensive overview of various techniques and advancements in detecting global anomalies in time series data by means of self-supervised learning.

### Self-Predictive Methods

- Anomaly detection on electroencephalography with self-supervised learning (2020) [[pdf]](https://ieeexplore.ieee.org/document/9313163)
- Water leak detection using self-supervised time series classification (2021) [[pdf]](https://www.sciencedirect.com/science/article/pii/S002002552100606X)
- Self-supervised Complex Network for Machine Sound Anomaly Detection (2021) [[pdf]](https://ieeexplore.ieee.org/document/9615923)
- Self-supervised anomaly detection via neural autoregressive flows with active learning (2021) [[pdf]](https://openreview.net/forum?id=LdWEo5mri6)
- Task-oriented self-supervised learning for anomaly detection in electroencephalography (2022) [[pdf]](https://arxiv.org/abs/2207.01391) [[code]](https://github.com/ironing/eeg-ad)
- OCSTN: One-class time-series classification approach using a signal transformation network into a goal signal (2022) [[pdf]](https://haie-lerco.cz/wp-content/uploads/2024/03/Hayashi_2022.pdf) [[code]](https://github.com/ToshiHayashi/OCSTN)
- Identification of High Emission Mobile Sources Based on Self-supervised Representation Network (2022) [[pdf]](https://link.springer.com/chapter/10.1007/978-981-19-9195-0_34)
- Adaptive memory networks with self-supervised learning for unsupervised anomaly detection (2022) [[pdf]](https://arxiv.org/abs/2201.00464) [[code]](https://github.com/zhangyuxin621/AMSL)
- Timeautoad: Autonomous anomaly detection with self-supervised contrastive loss for multivariate time series (2022) [[pdf]](https://ieeexplore.ieee.org/document/9705079)
- SSDPT: Self-supervised dual-path transformer for anomalous sound detection (2023) [[pdf]](https://arxiv.org/abs/2208.03421) [[code]](https://github.com/JishengBai/SSDPT)



### Contrastive Methods

- Neural transformation learning for deep anomaly detection beyond images (2021) [[pdf]](https://arxiv.org/abs/2103.16440) [[code]](https://github.com/boschresearch/NeuTraL-AD)
- Self-supervised contrastive pre-training for time series via time-frequency consistency (2022) [[pdf]](https://arxiv.org/abs/2206.08496) [[code]](https://github.com/mims-harvard/TFC-pretraining)
- Deep Contrastive One-Class Time Series Anomaly Detection (2023) [[pdf]](https://arxiv.org/abs/2207.01472) [[code]](https://github.com/ruiking04/COCA)

### Self-Predictive + Contrastive Methods

- Data Augmentation Based on Contrastive Learning in Audio Anomaly Detection (2022) [[pdf]](https://iopscience.iop.org/article/10.1088/1742-6596/2414/1/012011)
- Self-supervised acoustic anomaly detection via contrastive learning (2022) [[pdf]](https://ieeexplore.ieee.org/document/9746207) [[code]](https://github.com/Armanfard-Lab/AADCL)
- Joint Generative-Contrastive Representation Learning for Anomalous Sound Detection (2023) [[pdf]](https://arxiv.org/abs/2305.12111)


## 📊 Datasets for Time Series Anomaly Detection
In this section, we compile some of the most frequently used datasets for time series anomaly detection. These datasets are essential for benchmarking and evaluating the performance of various anomaly detection algorithms. They provide a standardized way to compare different approaches and ensure the reproducibility of experimental results. Each entry includes the name of the dataset, the type of anomalies considered (either local or global), a brief description of the dataset, the number of dimensions or variables in the time series (1 if univariate), and the reference of the paper in which the dataset was presented.
| Name                                    | Anomaly Type | Description                                                                                         | Num of Dims  | Reference
|-----------------------------------------|:----------:|-------------------------------------------------------------------------------------------------|:----------:|:----------:|
| [Yahoo-TSA](https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70)  | Local   | This dataset contains real and synthetic time series data with tagged anomalies, including outliers and change-points, representing various Yahoo service metrics and synthetic data with different trends, noise levels, and seasonalities​.             | 1      | [[1]](#ref1) |
| [SMAP](https://nsidc.org/data/smap/data) | Local     | This dataset contains telemetry data and soil samples from NASA's Soil Moisture Active Passive satellite mission. It is widely used for unsupervised anomaly detection tasks in time series data, featuring various telemetry channels with both normal and anomalous behavior recorded during the mission​. | 25      | [[2]](#ref2) |
| [SMD](https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset) | Local     | This dataset consists of 5 weeks of data collected from 28 different machines in a large internet company. It includes multiple dimensions and is used for training and testing anomaly detection algorithms in time series data​. | 38   | [[3]](#ref3) |
| [SWaT](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/) | Local     | This dataset contains data from a water treatment testbed, including both normal and attack conditions. It is widely used for research in anomaly detection in industrial control systems and consists of multivariate time series data collected over 11 days: 7 days of normal operation followed by 4 days of operation under 36 different attack scenarios​. | 51      | [[4]](#ref4) |
| [Tennessee Eastman Process](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1)| Local  | This dataset contains simulations of a large-scale chemical plant, featuring 52 variables and multiple fault conditions for benchmarking anomaly detection methods.                                         | 52| [[5]](#ref5) |
| [MSL](https://pds-atmospheres.nmsu.edu/data_and_services/atmospheres_data/Mars/Mars.html) | Local     | This dataset contains telemetry data and anomalies recorded from the Mars Science Laboratory rover. It includes various channels of time series data representing the rover's operational metrics, with both normal and anomalous states used for evaluating and benchmarking anomaly detection algorithms​. | 55      | [[2]](#ref2) |
| [WaDi](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/) | Local     | This dataset is an extension of the SWaT (Secure Water Treatment) system and is used to simulate and monitor the operations of a water distribution network. It includes multivariate time series data with normal and attack scenarios, making it valuable for testing anomaly detection methods in industrial control systems​. | 123      |  [[6]](#ref7) |
| [MIMII](https://zenodo.org/record/4740355) | Global     | This dataset contains audio recordings of industrial machines, specifically valves, pumps, fans, and slide rails. Each type of machine has multiple individual product models, and the dataset includes both normal and anomalous sounds to simulate real-life scenarios. Background noise recorded from real factories is mixed with the machine sounds to enhance realism, making it suitable for developing and testing sound-based anomaly detection methods​. | 1      |  [[7]](#ref8) |
| [ToyADMOS](https://zenodo.org/records/3351307#.XT-JZ-j7QdU) | Global     | This dataset consists of approximately 540 hours of normal machine operating sounds and over 12,000 samples of anomalous sounds. Collected with four microphones at a 48kHz sampling rate, the dataset was prepared by Yuma Koizumi and colleagues at NTT Media Intelligence Laboratories. It is designed for anomaly detection in machine operating sounds (ADMOS) research and includes three tasks: product inspection (toy car), fault diagnosis for fixed machines (toy conveyor), and fault diagnosis for moving machines (toy train). The dataset simulates real-world scenarios by deliberately damaging machine components to generate anomalous sounds​. | 1      |  [[8]](#ref9) |
| [UPenn and Mayo Clinic’s Seizure](https://www.kaggle.com/datasets/nxthuan512/upennmayoeegdatadog1) | Global     | This dataset is part of the Seizure Detection Challenge and includes intracranial EEG recordings from epilepsy patients. It contains both ictal (seizure) and interictal (non-seizure) segments from various patients, making it suitable for developing and testing algorithms for seizure detection. The dataset is designed to simulate real-world clinical conditions and is widely used for benchmarking seizure detection models in medical research. | 16      |  [[9]](#ref10) |
| [UCR Time Series Classification Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) | Global     | This archive has become a crucial resource in the time series data mining community. It initially contained 16 datasets and has expanded significantly over time, with the latest major update in 2015 increasing the collection to 128 datasets. These datasets are widely used for benchmarking and evaluating time series classification algorithms across various domains​. Unbalanced classification datasets are often employed for anomaly detection by designating the majority class as the normal category and categorizing the remaining classes as anomalous.| -      |  [[10]](#ref11) |
| [UCR Anomaly Archive](https://wu.renjie.im/research/anomaly-benchmarks-are-flawed/) | Local + Global     | This archive comprises 250 univariate time series dataset collected from various domains such as human medicine, biology, meteorology, and industry. While some datasets includes some natural anomalies, the majority are artificially injected to provide controlled conditions for testing anomaly detection algorithms. | -      | [[11]](#ref12) |

## 🔖 References

1. <a id="ref1"></a>Laptev, N., Amizadeh, S., & Billawala, Y. (2015). A benchmark dataset for time series anomaly detection. von Yahoo Research.
2. <a id="ref2"></a>Hundman, K., Constantinou, V., Laporte, C., Colwell, I., & Soderstrom, T. (2018, July). Detecting spacecraft anomalies using lstms and nonparametric dynamic thresholding. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining (pp. 387-395).
3. <a id="ref3"></a>Su, Y., Zhao, Y., Niu, C., Liu, R., Sun, W., & Pei, D. (2019, July). Robust anomaly detection for multivariate time series through stochastic recurrent neural network. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining (pp. 2828-2837).
4. <a id="ref4"></a>Mathur, A. P., & Tippenhauer, N. O. (2016, April). SWaT: A water treatment testbed for research and training on ICS security. In 2016 international workshop on cyber-physical systems for smart water networks (CySWater) (pp. 31-36). IEEE.
5. <a id="ref5"></a>Rieth, C. A., Amsel, B. D., Tran, R., & Cook, M. B. (2017). Additional tennessee eastman process simulation data for anomaly detection evaluation. Harvard Dataverse, 1, 2017.
6. <a id="ref7"></a>Ahmed, C. M., Palleti, V. R., & Mathur, A. P. (2017, April). WADI: a water distribution testbed for research in the design of secure cyber physical systems. In Proceedings of the 3rd international workshop on cyber-physical systems for smart water networks (pp. 25-28).
7. <a id="ref8"></a>Purohit, H., Tanabe, R., Ichige, K., Endo, T., Nikaido, Y., Suefusa, K., & Kawaguchi, Y. (2019). MIMII Dataset: Sound dataset for malfunctioning industrial machine investigation and inspection. arXiv preprint arXiv:1909.09347.
8. <a id="ref9"></a>Koizumi, Y., Saito, S., Uematsu, H., Harada, N., & Imoto, K. (2019, October). ToyADMOS: A dataset of miniature-machine operating sounds for anomalous sound detection. In 2019 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA) (pp. 313-317). IEEE.
9. <a id="ref10"></a>Temko, A., Sarkar, A., & Lightbody, G. (2015, August). Detection of seizures in intracranial EEG: UPenn and Mayo Clinic's seizure detection challenge. In 2015 37th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC) (pp. 6582-6585). IEEE.
10. <a id="ref11"></a>Dau, H. A., Bagnall, A., Kamgar, K., Yeh, C. C. M., Zhu, Y., Gharghabi, S., ... & Keogh, E. (2019). The UCR time series archive. IEEE/CAA Journal of Automatica Sinica, 6(6), 1293-1305.
11. <a id="ref12"></a>Wu, R., & Keogh, E. J. (2021). Current time series anomaly detection benchmarks are flawed and are creating the illusion of progress. IEEE transactions on knowledge and data engineering, 35(3), 2421-2429.

