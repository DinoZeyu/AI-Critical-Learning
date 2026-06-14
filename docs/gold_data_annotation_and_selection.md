Gold Data Annotation and Selection
==================================

Purpose
-------

The goal of this step is to build a high-quality gold evaluator set from the
existing train clean data.

The gold data is not intended to be used for training. It is used as a trusted
evaluation/reference set. To avoid train/evaluator leakage, any image selected
as gold is copied into Image_Data/Gold_Data/<dataset>/ and its row is removed
from Image_Data/Train_Clean_Data/<dataset>/labels.csv.

The original train clean image files are not deleted. Only the train labels are
rewritten, and the original full labels file is backed up before rewriting.


Input Data
----------

The script reads datasets from:

Image_Data/Train_Clean_Data/<dataset>/

For this project, the two expected datasets are:

Image_Data/Train_Clean_Data/STL/
Image_Data/Train_Clean_Data/Flower_102/

Each dataset folder is expected to contain:

images/
labels.csv

The labels.csv file must include at least these columns:

index
relative_path
label
class_name

The relative_path column points from the dataset root to the image file. For
example:

images/train_00000.png


Prompt
------

The OpenAI prompt is stored at:

Prompt/Gold_Data_Pick.txt

The prompt asks the model to judge whether an image-label pair is suitable for
a high-quality gold reference set. For each image, the model returns a strict
JSON object with:

data_reliance
rationale
alternative_class

data_reliance is a score from 0.00 to 1.00. Higher means the image-label pair
is more reliable as gold data.


Annotation Step
---------------

The script sends each image-label pair to the OpenAI API using:

gpt-4o-mini

For each row in the source labels file, the script:

1. Reads the image from Train_Clean_Data/<dataset>/<relative_path>.
2. Encodes the image as a base64 data URL.
3. Sends the image, target label id, target label name, dataset name, and class
   list to the OpenAI API.
4. Parses the model JSON response.
5. Appends the annotation record to:

Image_Data/Gold_Data/<dataset>/annotations.jsonl

annotations.jsonl contains one JSON record per image. This file stores the
complete LLM scoring result, not only the final gold subset.

The annotation process is resumable. If the script is interrupted, running the
same command again will read the existing annotations.jsonl file, skip images
that already have annotations, and continue with the remaining images.

Do not use --overwrite unless you intentionally want to delete the previous
annotations and start over.


Selection Strategy
------------------

After all rows are annotated, the script selects gold data using a hybrid
strategy:

1. Class balance target:
   Select up to 20% of each class as gold data.

2. Quality threshold:
   Only select images with:

   data_reliance >= 0.93

3. Ranking within each class:
   Among images that pass the quality threshold, the script sorts by
   data_reliance and selects the top candidates for that class.

This means the final gold count may be lower than exactly 20% if a class does
not have enough high-confidence images. In that case, the shortfall is recorded
in the selection summary.


Output Files
------------

After running the script for STL, the expected output is:

Image_Data/Gold_Data/STL/
  annotations.jsonl
  labels.csv
  selection_summary.csv
  selection_summary.json
  images/

After running the script for Flower_102, the expected output is:

Image_Data/Gold_Data/Flower_102/
  annotations.jsonl
  labels.csv
  selection_summary.csv
  selection_summary.json
  images/


Gold labels.csv
---------------

Gold_Data/<dataset>/labels.csv is the evaluator label file.

It includes the copied gold image path and the original source path. Important
columns include:

relative_path
  Path to the copied gold image inside Gold_Data/<dataset>/.

original_relative_path
  Original path inside Train_Clean_Data/<dataset>/.

original_image_path
  Full original image path before copying.

label
  Class id.

class_name
  Class name.

data_reliance
  LLM score used for gold selection.

rationale
  Short model explanation for the score.

alternative_class
  The model's suggested alternative class if the original label looked wrong or
  ambiguous. This is null when the original label appears correct.


Train Labels After Split
------------------------

The script rewrites:

Image_Data/Train_Clean_Data/<dataset>/labels.csv

After rewriting, this labels.csv file no longer contains rows selected as gold.
Training code that reads Train_Clean_Data/<dataset>/labels.csv will therefore
not train on gold evaluator images.

The original full train labels file is backed up once at:

Image_Data/Train_Clean_Data/<dataset>/labels_before_gold_split.csv

The original image files under Train_Clean_Data/<dataset>/images/ are not
deleted. They remain available for traceability and recovery, but the selected
gold rows are no longer referenced by the train labels.csv file.


How To Run
----------

Run from the repository root:

python scripts/Data_Prepration/LLM_Select_Gold.py --dataset STL
python scripts/Data_Prepration/LLM_Select_Gold.py --dataset Flower_102

For a small API smoke test:

python scripts/Data_Prepration/LLM_Select_Gold.py --dataset STL --limit 20

When using --limit, the script only annotates a small number of images and will
not perform final gold selection unless --select-partial is also passed. This is
intended for debugging only.


API Key Safety
--------------

The OpenAI API key should be stored in:

.env

Example:

OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

.env is ignored by git and should not be committed. The repository keeps
.env.example as a safe template without any real key.


Important Notes
---------------

This process uses the OpenAI API, so it does not require a Superpod GPU. Runtime
depends mainly on the number of images, network/API latency, and OpenAI rate
limits.

The main cost is API usage. The script may take many hours for a full dataset.

If a run is interrupted, rerun the same command without --overwrite. The script
will continue from the existing annotations.jsonl file.

The selection threshold can be changed at runtime if needed:

python scripts/Data_Prepration/LLM_Select_Gold.py --dataset STL --quality-threshold 0.90

The default threshold in the script is 0.93.
