# ADL22-HW3

## Dataset
[download link](https://drive.google.com/file/d/186ejZVADY16RBfVjzcMcz9bal9L3inXC/view?usp=sharing)

## Installation
```
/* create conda environment */
conda create --name <env_name> python=3.9
conda activate <env_name>

/* install tw_rouge */
pip install -e tw_rouge

/* install transformers */
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout t5-fp16-no-nans
pip install -e .

/* install other packages */
pip install -r requirements.txt
```

## Download
```
bash download.sh
```

## Usage
### Use the Script
```
usage: eval.py [-h] [-r REFERENCE] [-s SUBMISSION]

optional arguments:
  -h, --help            show this help message and exit
  -r REFERENCE, --reference REFERENCE
  -s SUBMISSION, --submission SUBMISSION
```

Example:
```
python eval.py -r public.jsonl -s submission.jsonl
{
  "rouge-1": {
    "f": 0.21999419163162043,
    "p": 0.2446195813913345,
    "r": 0.2137398792982201
  },
  "rouge-2": {
    "f": 0.0847583291303246,
    "p": 0.09419044877345074,
    "r": 0.08287844474014894
  },
  "rouge-l": {
    "f": 0.21017939117006337,
    "p": 0.25157090570020846,
    "r": 0.19404349000921203
  }
}
```

### Training
```
bash train.sh path/to/train.jsonl path/to/valid.jsonl
```
The training result will be at `mt5-small` directory.

### Testing
```
bash run.sh path/to/test.jsonl path/to/output.jsonl
```

## Reference
[cccntu/tw_rouge](https://github.com/cccntu/tw_rouge)
