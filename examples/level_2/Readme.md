## PromethAI Memory Manager



### Description


Initial code lets you do three operations:

1. Add to memory
2. Retrieve from memory
3. Structure the data to schema and load to duckdb

#How to use

## Installation

```docker compose build promethai_mem   ```

## Run

```docker compose up promethai_mem   ```


## Usage

The fast API endpoint accepts prompts and PDF files and returns a JSON object with the generated text.

```curl                                                                    
    -X POST                                                                                             
    -F "prompt=The quick brown fox"                                                                     
    -F "file=@/path/to/file.pdf"                                                                       
    http://localhost:8000/generate/                                                                    
```