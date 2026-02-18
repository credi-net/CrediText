<div align="center">


# CrediText

<img src="img/creditext.png" alt="CrediText Logo" style="width: 100px; height: auto;" />

Text extraction from Common Crawl raw files or web scraping, and production of text embeddings from this content to augment the graph data with text for GNNs.

Our optimized pipeline for extracting representative textual content from each monthly graph snapshot is derived from web crawls, and generates embeddings for downstream tasks. It leverages the Common Crawl monthly snapshot index to avoid scanning and downloading all Wet files data. The pipeline processes the snapshot WET files to extract textual content and stores it in an indexed, columnar Parquet format, enabling fast and parallelized LLM embedding generation. For our MLP labelled datsets  domain's lacking textual content, we employ a multi-threaded online scraping pipeline in batches to extract text directly from the domain’s home page.
</div>


## Getting Started

### Prerequisites

The project uses [uv](https://docs.astral.sh/uv/) to manage and lock project dependencies for a consistent and reproducible environment. If you do not have `uv` installed on your system, visit [this page](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

**Note**: If you have `pip`, you can invoke:

```sh
pip install uv
```

### Installation

```sh
# Clone the repo

# Enter the repo directory
cd CrediText

# Install core dependencies into an isolated environment
uv sync

# The isolated env is .venv
source .venv/bin/activate
```

## Usage
- Month: e.g. `Dec2024.txt` is a `.txt` file contains the common crawl month code e.g. ccmain202451
- start_idx: the start index inclusive to process out of 90K wet files
- end_idx: the last index inclusive to process out of 90K wet files
- seed_domains_list: the seed list of domains to extract thier content e.g. the dqr domains list at 'data/dqr/domain_pc1.csv'
- spark_table_name: the spark table name and output folder naming pattern i.e, content_table
- WetFilesOrder: **Optional** The order of which the common Crawl WET files are downloaded and processed starting from start_idx and ending with the end_idx

### Build Monthly domains Index 
Extract month doamins and thier corssponding WET Files for parallel and ordered content extraction
```sh
cd bash_scripts
./end-to-end.sh  <Month> <start_idx> <end_idx> [cc-index-table] <seed_domains_list.csv>
e.g. 
./end-to-end.sh  CC-Crawls/Dec2024.txt 0 10  [cc-index-table] ../data/Dec2024/Dec2024_domains.csv
```
### Extract Montly domain's text content

```sh
cd bash_scripts
bash end-to-end.sh <Month> <start_idx> <end_idx> [wet] <seed_domains_list.csv> <temp_spark_table_name> <WetFilesOrder.txt>
e.g. 
bash end-to-end.sh  CC-Crawls/Feb2025.txt 0 10 [wet] ../data/Dec2024/Dec2024_domains.csv content_ext_table spark-warehouse/creditext_ccmain202451_wetFilesOrder.txt
```
This will generate parquet files per batch under \`$SCRATCH/spark-warehouse/\<spark_table_name>_batch_ccmain202451_\<start_idx>\_\<end_idx>

### Merging the extracted Content

Loop across the generated parquet files to collect text contents and merge them per domain\
simply use:  `pandas.read_parquet(file_path, engine='pyarrow')`\
Each parquet file contains columns:
- Domain_Name: the domain name
- WARC_Target_URI: the web page URI
- WARC_Identified_Content_Language: list of CC-identified content languagues
- WARC_Date: the content scrap date
- Content_Type: the content type i.e., text,csv,json
- Content_Length: the content length in bytes
- wet_record_txt: the UTF-8 text content

______________________________________________________________________

### Generate Content Embedding 

### - use huggingface ready content files

``` python
uv run python creditext/content_embbeding/generate_content_embedding.py.py  --hf_files_start_idx=0 --hf_files_end_idx=10 --parquet_batch_size=100000 --emb_batch_size=5000 --emb_dim=256 --parquet_start_batch_idx=0 --hf_repo_id=Hussein-Abdallah/CrediBench-WebContent-<Month>
```
### - use your extracted files
``` python
uv run python creditext/content_embbeding/generate_content_embedding.py.py  --hf_files_start_idx=0 --hf_files_end_idx=10 --parquet_batch_size=100000 --emb_batch_size=5000 --emb_dim=256 --parquet_start_batch_idx=0 --local_dir=<your_extracted_content_parquet_files_ path>
```
______________________________________________________________________

### Running MLP Experiments
#### DQR regression experiments

The required embedding files and dataset must be placed under the data directory. The expected file paths are as follows:

- DQR domain text embeddings: data/dqr/dqr_text_<emb_model>_<emb_dim>.pkl e.g. dqr_dec_text_embeddinggemma-300m_256.pkl
- DQR dataset (ratings): data/dqr/domain_ratings.csv

```sh
uv run python creditext/experiments/mlp_experiments/mlp_train_regressor.py --dqr_target pc1 --embed_type text --emb_model embeddinggemma-300m
```
**review the paramters options to reproduce all experiments**

#### DomainRel Binary Classifcation experiments

The required embedding files and dataset must be placed under the data directory. The expected file paths are as follows:

- DomainRel text embeddings: data/weaksupervision/weak_content_emb_<month>_<emb_model>_<emb_dim>.parquet e.g. weak_content_emb_dec2024_embeddinggemma-300m_256.parquet
- DomainRel dataset (labels): data/weaksupervision/weaklabels.csv

```sh
uv run python creditext/experiments/mlp_experiments/mlp_train_classifier.py --domainRel_target weaklabel --embed_type text --emb_model embeddinggemma-300m
```
**review the paramters options to reproduce all experiments**

To learn more about making a contribution to CrediTest see our [contribution guide](./.github/CONTRIBUTION.md)

______________________________________________________________________

### Citation

```
@article{kondrupsabry2025credibench,
  title={{CrediBench: Building Web-Scale Network Datasets for Information Integrity}},
  author={Kondrup, Emma and Sabry, Sebastian and Abdallah, Hussein and Yang, Zachary and Zhou, James and Pelrine, Kellin and Godbout, Jean-Fran{\c{c}}ois and Bronstein, Michael and Rabbany, Reihaneh and Huang, Shenyang},
  journal={arXiv preprint arXiv:2509.23340},
  year={2025},
  note={New Perspectives in Graph Machine Learning Workshop @ NeurIPS 2025},
  url={https://arxiv.org/abs/2509.23340}
}
```
