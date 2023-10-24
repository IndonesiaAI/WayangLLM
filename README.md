# WayangLLM

## Hugging Face Datasets
1. [Raw StackExchange HTML](https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences)
2. [Processed ST Pairs](https://huggingface.co/datasets/lvwerra/stack-exchange-paired)
3. [ST Pair Splits](https://huggingface.co/datasets/IndonesiaAI/stackexchange-paired-part-0-revision)
4. [Cleaned Data Splits](https://huggingface.co/datasets/IndonesiaAI/cleaned-data-split-0)

Note: change `part-id` to access different splits. 

## Data Cleaning Process

### Raw HTML Data Cleaning
1. **Regex Filtering**: Utilizes regular expressions to remove code and math tags from the raw HTML data.
2. **QID Extraction**: Extracts and saves filtered Question IDs (QIDs).

### ST Pairs Refinement
1. **QID Matching**: Matches QIDs in ST pairs.
2. **Language Source Exclusion**: Filters out data from specific StackExchange communities based on a given list. For reference, check [StackExchange Sites](https://stackexchange.com/sites).
3. **Further Cleaning**: Additional regex-based cleaning to remove math (`$...$`) and code (```...```) elements.
   
