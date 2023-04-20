# AD prediction from exRNA in blood

The goal of this project is to predict Alzheimer Disease using blood extracellular RNA. We have 3 studies available:

- Data from our SILVER-seq technique.
- Data from [Toden et al.](https://advances.sciencemag.org/content/6/50/eabb1654) published on Bioproject at [this link](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA574438/).
- Data from [Burgos et al.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0094839) uploaded on dbGaP and downloadable under official request.

We aim to obtain a gene set that can be used together with personal information that are shared among the cohorts, such as sex, age, ApoE status, etc. to increase the power of the prediction. We might decide to discard some samples if any of the features chosen in our model is not available for them. 

Data were processed consistenly using STAR as aligner with `--sjdbGTFfile Homo_sapiens.GRCh38.84.chr.gtf` and then featureCounts to count uniquely mapped reads (MAPQ = 255) over exons. Gene IDs are consistent among the 3 datasets and are ENSEMBL IDs.

Per each dataset we provide:

- Metadata with sample information.
- Raw count table where genes are on the rows and samples are on the columns.

## SILVER-seq

Converter samples are available but not considered here.

115 samples in total:

- 41 Normal (N) from 9 donors
- 74 Alzheimer Disease (AD) from 15 donors

Data:

- [Metadata](./silver_seq/silver_seq_metadata.xlsx)
- [Count table](./silver_seq/silver_seq_counts.txt)

## Toden et al.

One sample from the pool is discarded being control water. Other 3 samples of patient 1031 are discarded because this one appears as a converter (1 sample NCI, 2 samples AD).

334 samples in total:

- 164 Normal (NCI) from 114 donors
- 170 Alzheimer Disease (AD) from 126 donors

Data:

- [Metadata](./toden/toden_metadata.xlsx)
- [Count table](./toden/toden_counts.txt)

## Burgos et al.

267 samples in total:

- 138 Normal (Control) from 74 donors
- 129 Alzheimer Disease (Alzheimer's Disease) from 70 donors

Data:

- [Metadata](./burgos_dbgap/burgos_dbgap_metadata.xlsx)
- [Count table](./burgos_dbgap/burgos_dbgap_counts.txt)


